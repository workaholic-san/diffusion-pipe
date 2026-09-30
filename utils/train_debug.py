"""Optimizer diagnostics for train.py.

Wraps `optimizer.step` and records norms that the Muon/AdamW split makes
otherwise invisible: which group a parameter landed in, how far it moved, and
what the AdamW state tensors look like.

Three deliberate differences from the Colab version this was ported from:

* **No fabricated zeros.** AdamW metrics are only emitted when a group actually
  carries `use_muon is False`. The old version logged `0.0` for every
  `debug/adamw/*` key under any other optimizer, which reads as a measurement
  of zero rather than as "not applicable".
* **One device sync per metric, not one per tensor.** Norms are accumulated on
  the device and read back once at the end, instead of calling `.item()` inside
  the per-tensor loop.
* **The cadence defaults to 10, not to `logging_steps`.** `logging_steps`
  defaults to 1 in this repo, so tying the debug cadence to it silently ran
  every metric on every training step.
"""

import math
import types


def _is_main_process():
    """Rank check that does not import deepspeed (see utils/lr_schedulers.py)."""
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def _group_label(index, group):
    if group.get('use_muon') is True:
        return 'muon'
    if group.get('use_muon') is False:
        return 'adamw'
    return f'group{index}'


def _has_adamw_group(optimizer):
    return any(g.get('use_muon') is False for g in optimizer.param_groups)


def _collect_pre_step(optimizer, want):
    """Accumulate pre-step norms on-device. Returns a dict of device tensors."""
    totals = {}
    adamw_params = []

    for gi, group in enumerate(optimizer.param_groups):
        label = _group_label(gi, group)
        is_adamw = group.get('use_muon') is False

        if is_adamw and want['update_norm']:
            for p in group['params']:
                adamw_params.append(p.detach().clone())

        grad_sq = None
        param_sq = None
        do_param = (label == 'muon' and want['muon_param_norm']) or \
                   (label == 'adamw' and want['adamw_param_norm'])

        for p in group['params']:
            if want['grad_norm'] or want['group_grad_norm']:
                if p.grad is not None:
                    g2 = p.grad.detach().float().pow(2).sum()
                    if want['grad_norm']:
                        totals['global'] = totals.get('global', 0) + g2
                    if want['group_grad_norm']:
                        grad_sq = g2 if grad_sq is None else grad_sq + g2
            if do_param:
                s = p.detach().float().pow(2).sum()
                param_sq = s if param_sq is None else param_sq + s

        if want['group_grad_norm'] and grad_sq is not None:
            totals[f'group_{gi}_{label}_grad'] = grad_sq
        if do_param and param_sq is not None:
            totals[f'group_{gi}_{label}_param'] = param_sq

    return totals, adamw_params


def install_debug_hook(optimizer, config, model_engine):
    """Attach the diagnostic step wrapper. Returns the resolved config."""
    debug_cfg = config.get('debug', {}) if isinstance(config, dict) else {}

    # 10, not logging_steps: logging_steps is 1 by default here, and computing
    # these norms every step costs a device sync per step.
    every = int(debug_cfg.get('debug_every_n_steps', 10) or 10)
    if every < 1:
        every = 1

    want = {
        'grad_norm': bool(debug_cfg.get('log_grad_norm', True)),
        'group_grad_norm': bool(debug_cfg.get('log_group_grad_norm', True)),
        'muon_param_norm': bool(debug_cfg.get('log_muon_param_norm', False)),
        'adamw_param_norm': bool(debug_cfg.get('log_adamw_param_norm', True)),
        'update_norm': bool(debug_cfg.get('log_adamw_update_norm', True)),
        'state_norm': bool(debug_cfg.get('log_adamw_state_norm', True)),
        'every': every,
    }
    # Only ask for AdamW-side metrics when an AdamW group exists at all.
    if not _has_adamw_group(optimizer):
        want['adamw_param_norm'] = False
        want['update_norm'] = False
        want['state_norm'] = False

    if _is_main_process() and bool(debug_cfg.get('log_param_groups', True)):
        print('\nOptimizer param_groups:')
        for i, group in enumerate(optimizer.param_groups):
            info = {k: v for k, v in group.items() if k != 'params'}
            print(f'  group {i} ({len(group["params"])} params): {info}')

    original_step = optimizer.step

    def _patched_step(self, *args, **kwargs):
        cfg = getattr(self, '_debug_config', want)
        if getattr(self, '_debug_step', None) is None:
            try:
                self._debug_step = int(getattr(model_engine, 'global_steps', 0))
            except Exception:
                self._debug_step = 0
        self._debug_step += 1
        current = self._debug_step

        if current % cfg['every'] != 0:
            return original_step(*args, **kwargs)

        metrics = {}
        pre = []
        try:
            totals, pre = _collect_pre_step(self, cfg)
        except Exception as exc:
            if _is_main_process():
                print(f'debug pre-step failed: {exc}')
            totals, pre = {}, []

        # Deliberately outside the try: a real optimizer failure must propagate
        # rather than be swallowed by the diagnostics.
        result = original_step(*args, **kwargs)

        try:
            for key, device_scalar in totals.items():
                value = float(device_scalar.sqrt().item())
                if key == 'global':
                    metrics['train/grad_norm'] = value
                    model_engine._custom_global_grad_norm = value
                    model_engine._custom_global_grad_norm_step = current
                elif key.endswith('_grad'):
                    metrics[f'debug/{key.replace("_grad", "")}/grad_norm'] = value
                else:
                    metrics[f'debug/{key.replace("_param", "")}/param_norm'] = value

            if cfg['update_norm'] and pre:
                # Re-pair by position: pre holds a clone per AdamW param in
                # group order, and nothing below reorders param_groups.
                idx = 0
                delta_sq = 0.0
                for group in self.param_groups:
                    if group.get('use_muon') is not False:
                        continue
                    for p in group['params']:
                        if idx < len(pre):
                            delta_sq += float(
                                (p.detach().float() - pre[idx].float()).pow(2).sum().item())
                        idx += 1
                metrics['debug/adamw/update_norm'] = math.sqrt(delta_sq)

            if cfg['state_norm']:
                for name in ('exp_avg', 'exp_avg_sq', 'kahan_comp'):
                    acc = 0.0
                    for group in self.param_groups:
                        if group.get('use_muon') is not False:
                            continue
                        for p in group['params']:
                            st = self.state.get(p, {})
                            if name in st:
                                acc += float(st[name].float().pow(2).sum().item())
                    metrics[f'debug/adamw/{name}_norm'] = math.sqrt(acc)

            self._debug_metrics = metrics
            self._debug_metrics_step = current
        except Exception as exc:
            if _is_main_process():
                print(f'debug post-step failed: {exc}')

        return result

    optimizer.step = types.MethodType(_patched_step, optimizer)
    optimizer._debug_config = want
    if _is_main_process():
        print(f'Debug hook installed (every {every} steps)')
    return want
