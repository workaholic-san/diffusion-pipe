"""Muon optimizer with a Kahan-summation AdamW fallback group.

What upstream already does (verified against pytorch-optimizer 3.10.1)
-------------------------------------------------------------------------
`pytorch_optimizer.Muon` is not a plain Muon. Read from its source:

* Its `__init__` REFUSES to build without a `use_muon` key on every group
  (`ValueError: 'use_muon' must be set.`), so the caller owns the split. It
  already accepts `adamw_lr`, `adamw_betas`, `adamw_eps` and `adamw_wd` as
  defaults for the non-Muon groups, and it fills each group from those via
  `group.get(key, default)` — per-group values win.
* Its `step()` already branches on `group['use_muon']`, orthogonalizes matrix
  updates with Newton-Schulz, and flattens `update.ndim > 2` to 2-D.
* Its AdamW branch writes `p.addcdiv_(...)` straight into the parameter.

So the split itself is NOT the value added here; that already exists upstream.
What this module replaces is exactly one thing: the AdamW branch. Upstream
writes the update directly into the (usually bf16) parameter, so any update
smaller than half a bf16 ulp rounds straight back to the same value and is
silently discarded. `MuonKahan` keeps the upstream Muon path untouched and
substitutes a compensated AdamW for everything else.

Why Kahan
----------
bf16 keeps 8 mantissa bits. At a parameter magnitude of ~0.02 one ulp is
~1.2e-4, while a typical AdamW step is ~1e-6 — about 1% of an ulp. A naive
write therefore rounds to "no change" and the update is lost. This module
keeps the exact running total in fp32 (`kahan_comp`) and only moves `p` once
the accumulated total crosses a rounding boundary.

Group contract
--------------
`split_param_groups` tags every group it produces with:
  * `use_muon`: True  -> stepped by the upstream Muon implementation
                 False -> stepped by `_adamw_kahan_step` in this module
                 (absent -> upstream raises, which is intentional)
  * `kahan`:   bool  -> whether to carry the fp32 compensation tensor
  * `betas` / `eps`: AdamW hyperparameters for the non-Muon group only

Per-group learning rates
------------------------
`adamw_lr` sets ONE learning rate for every AdamW group. When a model hands
its groups a `group_name` (anima does: `base`, `self_attn`, `cross_attn`,
`mlp`, `mod`, `llm_adapter` — see `models/cosmos_predict2.py`), the config can
address them individually:

    adamw_group_lr = { mod = 8e-5, self_attn = 1.5e-4 }

The name belongs to the model, not to this module: a group without
`group_name`, or one whose name is not in the table, keeps `adamw_lr`. The
override is deliberately restricted to AdamW groups — the Muon groups keep the
lr their own group carries. A table key that matches no AdamW group is
reported at startup rather than silently doing nothing, because a typo there
looks exactly like a learning rate that was never applied.

Supported versions
------------------
Verified against **pytorch-optimizer 3.10.1**, the version installed here and
the one whose `Muon` source was read to establish the contract above.
`requirements.txt` leaves the dependency unpinned, so a future resolution may
be a different Muon implementation; the assumptions this subclass relies on are
the three named in the first section (per-group `use_muon`, the `adamw_*`
defaults, and an AdamW branch that writes straight into the parameter). A
version that breaks any of them is not supported by this module, and the
`use_muon` requirement surfaces as an immediate ValueError rather than as a
quietly wrong update.

Configuration is consumed in `train.get_optimizer`, which pops the
`muon_*` / `adamw_*` keys out of the shared kwargs dict so they can never reach
an unrelated optimizer's constructor.
"""

import math

import torch

try:  # pytorch_optimizer is an optional import: keep this module importable without it
    from pytorch_optimizer import Muon as _BaseMuon
    _MUON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on a broken install
    _MUON_IMPORT_ERROR = exc

    class _BaseMuon:
        """Placeholder base that fails loudly instead of silently doing nothing.

        Degrading to `object` here would produce a MuonKahan that constructs
        fine, accepts no arguments, and silently applies no Muon update at all —
        a training run that looks healthy and is not. A missing dependency must
        be a loud failure.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'pytorch_optimizer is required by muon_kahan but could not be '
                f'imported: {_MUON_IMPORT_ERROR!r}. Install it with '
                '`pip install pytorch-optimizer` (it is listed in '
                'requirements.txt).'
            ) from _MUON_IMPORT_ERROR


#: Config keys owned by the Muon/AdamW split. `train.get_optimizer` pops these
#: unconditionally so that leaving them in a config while selecting another
#: optimizer can never raise `TypeError: unexpected keyword argument`.
MUON_SPLIT_KEYS = (
    'muon_min_size',
    'adamw_lr',
    'adamw_group_lr',
    'kahan_adamw',
    'adamw_betas',
    'adamw_eps',
    'adamw_weight_decay',
)


def _is_main_process():
    """Rank check that does not import deepspeed (see utils/lr_schedulers.py)."""
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def _adamw_group_lr_table(raw):
    """Validate the `group_name -> lr` table and return it with float values.

    A malformed entry has to fail here, loudly and once. Coerced lazily it
    would surface much later as a step that silently did nothing.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            'adamw_group_lr must be a table of group-name -> learning rate, '
            f'got {type(raw).__name__}')
    table = {}
    for name, value in raw.items():
        try:
            lr = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'adamw_group_lr[{name!r}] = {value!r} is not a number') from exc
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError(
                f'adamw_group_lr[{name!r}] must be a positive finite learning '
                f'rate, got {value!r}')
        table[str(name)] = lr
    return table


def is_muon_param(param, muon_min_size=32):
    """A parameter goes to Muon when it is a matrix big enough to orthogonalize.

    For a 4-D conv weight the shape is (out, in, *spatial). Orthogonalization
    acts on the flattened (out, in*spatial) matrix, so the test is made on those
    axes. Judging the trailing spatial axes instead would route a [128, 64, 3, 3]
    weight to AdamW purely because its kernel is 3x3, which is not what the
    Newton-Schulz step is for.
    """
    shape = getattr(param, 'shape', None)
    ndim = getattr(param, 'ndim', 1)
    if shape is None or ndim < 2:
        return False
    if ndim > 2:
        rows = shape[0]
        cols = 1
        for dim in shape[1:]:
            cols *= dim
        return min(rows, cols) >= muon_min_size
    return min(shape[-2:]) >= muon_min_size


def split_param_groups(param_groups, muon_min_size=32, adamw_lr=None,
                       adamw_group_lr=None, kahan_adamw=True, adamw_betas=(0.9, 0.999),
                       adamw_eps=1e-8, adamw_weight_decay=None):
    """Split every group into a Muon group and an AdamW group.

    `adamw_weight_decay=None` (the default) means "leave whatever the group
    already carries". That matters: `train.get_optimizer` deliberately sets
    `weight_decay = 0` on the group holding 1-D parameters and `llm_adapter.embed`,
    and an override applied unconditionally would silently undo that decision.

    `adamw_group_lr` overrides `adamw_lr` per named group (see the module
    docstring). A group with no `group_name` — every model that does not name
    its groups — keeps `adamw_lr`, so this stays a per-model opt-in.
    """
    lr_table = _adamw_group_lr_table(adamw_group_lr)
    matched = set()

    out = []
    for group in param_groups:
        muon_params = [p for p in group['params'] if is_muon_param(p, muon_min_size)]
        adamw_params = [p for p in group['params'] if not is_muon_param(p, muon_min_size)]

        if muon_params:
            g = group.copy()
            g['params'] = muon_params
            g['use_muon'] = True
            out.append(g)

        if adamw_params:
            g = group.copy()
            g['params'] = adamw_params
            g['use_muon'] = False
            name = g.get('group_name')
            if name is not None and name in lr_table:
                g['lr'] = lr_table[name]
                matched.add(name)
            elif adamw_lr is not None:
                g['lr'] = float(adamw_lr)
            if adamw_weight_decay is not None:
                g['weight_decay'] = float(adamw_weight_decay)
            g['betas'] = tuple(float(b) for b in adamw_betas)
            g['eps'] = float(adamw_eps)
            g['kahan'] = bool(kahan_adamw)
            out.append(g)

    unmatched = sorted(set(lr_table) - matched)
    if unmatched and _is_main_process():
        # Loud on purpose: a name that matches nothing (typo, or a category with
        # no AdamW parameters) leaves the learning rate exactly as it would have
        # been without the table, which is indistinguishable from "it had no
        # effect" until someone checks the numbers.
        print(f'Warning: adamw_group_lr names {unmatched} matched no AdamW '
              f'group; those learning rates were not applied. Group names seen: '
              f'{sorted({str(g.get("group_name")) for g in out})}')

    return out


class MuonKahan(_BaseMuon):
    """Muon for matrix params, Kahan-summation AdamW for everything else."""

    @torch.no_grad()
    def step(self, closure=None):
        muon_groups = [g for g in self.param_groups if g.get('use_muon')]
        adamw_groups = [g for g in self.param_groups if g.get('use_muon') is False]

        loss = None
        if muon_groups:
            # Temporarily narrow the group list the upstream step() iterates.
            # Muon and AdamW buckets are disjoint by construction, so the swap
            # cannot leave a parameter unstepped.
            old_groups = self.param_groups
            self.param_groups = muon_groups
            try:
                loss = super().step(closure)
            finally:
                self.param_groups = old_groups
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        if adamw_groups:
            self._adamw_kahan_step(adamw_groups)

        return loss

    def _adamw_kahan_step(self, groups):
        for group in groups:
            betas = group.get('betas', (0.9, 0.999))
            if isinstance(betas, (list, tuple)) and len(betas) == 2:
                beta1, beta2 = float(betas[0]), float(betas[1])
            else:
                beta1, beta2 = 0.9, 0.999

            eps = float(group.get('eps', 1e-8))
            if eps <= 0.0:
                # eps == 0 turns a zero gradient into a permanent NaN in the
                # state, which then poisons every later step.
                eps = 1e-8
            weight_decay = float(group.get('weight_decay', 0.0))
            lr = float(group['lr'])
            use_kahan = bool(group.get('kahan', True))

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    # torch's sparse path uses different math (untouched rows
                    # get no momentum), which this dense implementation cannot
                    # express. Densifying is exact and degrades one tensor
                    # instead of aborting a multi-day run.
                    grad = grad.coalesce().to_dense()

                state = self.state[p]

                # Keyed on 'step' rather than emptiness: a partially populated
                # state (e.g. restored from a DeepSpeed checkpoint) would
                # otherwise raise KeyError below.
                if 'step' not in state:
                    state['step'] = 0
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                if 'exp_avg_sq' not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                if use_kahan and 'kahan_comp' not in state:
                    state['kahan_comp'] = torch.zeros_like(p, dtype=torch.float32)

                # Guard against states restored with a narrower dtype.
                if state['exp_avg'].dtype != torch.float32:
                    state['exp_avg'] = state['exp_avg'].float()
                if state['exp_avg_sq'].dtype != torch.float32:
                    state['exp_avg_sq'] = state['exp_avg_sq'].float()
                if use_kahan and state['kahan_comp'].dtype != torch.float32:
                    state['kahan_comp'] = state['kahan_comp'].float()

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                grad_f = grad.float()

                exp_avg.mul_(beta1).add_(grad_f, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1.0 - beta2)

                bias1 = 1.0 - beta1 ** step
                bias2 = 1.0 - beta2 ** step

                # Same construction as torch.optim.AdamW: eps is added *after*
                # the bias2 division, and the out-of-place ops keep the stored
                # state buffers uncorrupted by the in-place `_` variants.
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
                update = (exp_avg / bias1).div_(denom)

                if use_kahan:
                    comp = state['kahan_comp']

                    # Reconstruct the exact running total from the *stored*
                    # parameter plus the residual. Doing this (instead of
                    # keeping a separate fp32 master) is what keeps p and the
                    # running total locked together: p + comp is invariant
                    # across the bf16 rounding step.
                    virtual = p.detach().float() + comp

                    # Decoupled weight decay on the fp32 virtual value, so the
                    # Kahan invariant survives it.
                    if weight_decay != 0.0:
                        virtual = virtual * (1.0 - lr * weight_decay)

                    target = virtual - update * lr
                    p_new = target.to(p.dtype)

                    # Overwrite (do not accumulate) the residual: this is the
                    # compensation for the single rounding just performed.
                    comp.copy_(target - p_new.float())
                    p.data.copy_(p_new)
                else:
                    # No compensation, but still one fp32 round trip so weight
                    # decay and the update are not rounded separately.
                    target = p.detach().float()
                    if weight_decay != 0.0:
                        target = target * (1.0 - lr * weight_decay)
                    target = target - update * lr
                    p.data.copy_(target.to(p.dtype))
