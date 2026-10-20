"""Collect clean teacher continuations after per-environment recovery roll-ins.

No simulator reset occurs between roll-in and teacher takeover. Scheduling uses
the simulation's push metadata solely to select the collection window; the
teacher and student receive current observation histories only. Original
collect.py remains unchanged.
"""
import json
import hashlib
import math
from pathlib import Path

import torch


DEFAULT_SCHEDULE = [
    dict(name='push_start', after_push_end_steps=-6),
    dict(name='push_end', after_push_end_steps=0),
    dict(name='push_end_plus6', after_push_end_steps=6),
    dict(name='push_end_plus12', after_push_end_steps=12),
]
COUNT_FIELDS = ('attempted', 'finished', 'takeovers', 'succeeded_after_takeover',
                'failed_before_takeover', 'completed_before_takeover',
                'failed_after_takeover', 'unresolved', 'saved')


def force_bucket(bw):
    for level in (0., .45, .75):
        if math.isclose(bw, level, abs_tol=1e-5):
            return f'{level:g}'
    return '0_to_0.45' if 0 < bw < .45 else '0.45_to_0.75' if bw < .75 else 'above_0.75'


def reset_controller(controller, ids=None, all_ids=None):
    """Honor stateful teachers without clearing other live slots.

    Older stateless/student controllers may only support full-batch reset().
    Their partial reset is skipped; no completed slot is recycled in a batch.
    """
    import inspect
    reset = getattr(controller, 'reset', None)
    if reset is None:
        return False
    signature = inspect.signature(reset)
    candidates = ([ ((), {}) ] if ids is None else [])
    selected_ids = all_ids if ids is None else ids
    candidates += [((selected_ids,), {}), ((), {'ids': selected_ids}), ((), {'env_ids': selected_ids})]
    for args, kwargs in candidates:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        reset(*args, **kwargs)
        return True
    if ids is None:
        raise TypeError('Controller reset signature supports neither full reset nor environment ids')
    return False


@torch.no_grad()
def check_causal_features(task, policy):
    """Change only future push metadata; current features must not change.

    This checks the actual feature computation (including helper calls), rather
    than relying on its name. It is a bounded invariance check, not a proof that
    every possible future source has been excluded. No physics step is taken.
    """
    before_history = task.history.clone()
    before = task.features().clone()
    names = ('strength', 'direction', 'start_step')
    saved = {name: getattr(task, name).clone() for name in names}
    try:
        task.strength.copy_(saved['strength'] + .137)
        task.direction.copy_(torch.roll(saved['direction'], 1, dims=-1) + .071)
        task.start_step.copy_(saved['start_step'] + 11)
        changed = task.features().clone()
    finally:
        for name, value in saved.items():
            getattr(task, name).copy_(value)
    if not torch.isfinite(before).all() or not torch.isfinite(changed).all():
        raise ValueError('Nonfinite features during causal-input check')
    difference = float((before - changed).abs().max())
    if not torch.allclose(before, changed, atol=1e-6, rtol=0):
        raise ValueError(f'Features depend on future perturbation metadata (max delta {difference})')
    if not torch.equal(task.history, before_history):
        raise ValueError('features() mutated observation history during causal-input check')
    # Do not call the teacher here: a hybrid/recurrent teacher may update its
    # internal state on every call. Actual labels query it once per physics step.
    return dict(checked_fields=list(names), max_feature_difference=difference,
                physics_steps_during_check=0,
                teacher_input='Current four-frame task.history only',
                perturbation_metadata_passed_to_teacher=False,
                schedule_metadata_used_only_by_collector=True,
                check_scope='Runtime feature invariance to force amplitude, direction and push start time')


@torch.no_grad()
def collect(task, policy, cfg, rollin_policy=None):
    """Called by the existing collect runtime with a validated clean teacher.

    cfg additions:
      rollin_method: noisyteacher (default), teacher, bc or diffusion
      takeover_schedule: list of {name, after_push_end_steps}, default -6/0/6/12
      rollin_student_checkpoint: required for bc/diffusion unless injected
      rollin_noise: normalized PD residual noise for noisyteacher only

    rollin_policy injection exists for simulator-free tests and explicit caller
    composition. A provided student must advertise its feature_version/kind.
    """
    if not all(row['split'] == 'train' for row in task.rows):
        raise ValueError('Recovery collection only accepts training sources')
    from motion_reference import file_sha256
    if not cfg.get('checkpoint'):
        raise ValueError('Warm correction collection requires a selected checkpoint artifact')
    checkpoint_sha256=file_sha256(cfg['checkpoint'])
    frozen_prior=dict(policy.prior.identity)
    provenance=dict(cfg.get('teacher_provenance') or {})
    if provenance.get('teacher_checkpoint_sha256') not in (None, checkpoint_sha256):
        raise ValueError('Selected checkpoint changed after the teacher was constructed')
    provenance.update(teacher_checkpoint_sha256=checkpoint_sha256, frozen_prior=frozen_prior,
                      action_target_version='executed_total_minus_composed_terrain_prior_v1')
    cfg['teacher_provenance']=provenance
    cfg['teacher_identity']='warm_collected:'+hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('episode_*.pt')):
        raise ValueError('Use a fresh output directory; existing episodes must not be overwritten')
    fixed_attempts = int(cfg.get('fixed_attempts', 0))
    if fixed_attempts and (fixed_attempts < task.n or fixed_attempts % task.n):
        raise ValueError('fixed_attempts must be a positive multiple of num_envs')
    desired = fixed_attempts or int(cfg.get('episodes', 2500))
    if desired < 1:
        raise ValueError('episodes must be positive')
    method = cfg.get('rollin_method', 'noisyteacher')
    if method not in ('teacher', 'noisyteacher', 'bc', 'diffusion'):
        raise ValueError(f'Unsupported roll-in method: {method}')
    feature_version = getattr(task, 'feature_version', 'v1_world_contact')
    if method in ('bc', 'diffusion'):
        if rollin_policy is None:
            from student_ema import StudentController
            rollin_policy = StudentController(
                cfg['rollin_student_checkpoint'], task.device,
                seed=cfg.get('seed', 17), execute_steps=cfg.get('rollin_execute_steps', 1),
                noise_mode=cfg.get('rollin_noise_mode', 'independent'),
                inference_steps=cfg.get('rollin_inference_steps', 8))
        if rollin_policy.feature_version != feature_version:
            raise ValueError('Roll-in student and task feature conventions differ')
        if rollin_policy.model.kind != method:
            raise ValueError('Roll-in checkpoint controller kind does not match rollin_method')
        if hasattr(task, 'gate_state'):
            from causal_gate_state import InactiveZeroStudent
            if not isinstance(rollin_policy, InactiveZeroStudent):
                rollin_policy = InactiveZeroStudent(rollin_policy, task)
    elif rollin_policy is not None:
        raise ValueError('Injected policy is only accepted for bc/diffusion roll-ins')
    pulse_steps = int(cfg.get('push_duration_steps', 6))
    # The current RecoveryTask applies the force for exactly six control steps.
    if pulse_steps != 6:
        raise ValueError('Current RecoveryTask uses a six-step pulse; do not silently change scheduling')
    schedule = cfg.get('takeover_schedule', DEFAULT_SCHEDULE)
    if not schedule or len({s['name'] for s in schedule}) != len(schedule):
        raise ValueError('takeover_schedule needs unique names and at least one entry')
    if any(int(s['after_push_end_steps']) < -pulse_steps for s in schedule):
        raise ValueError('Takeover is limited to push start or later')
    storage_dtype = {'float16': torch.float16, 'float32': torch.float32}[cfg.get('storage_dtype', 'float16')]
    max_attempts = fixed_attempts or int(cfg.get('max_attempts', max(desired * 5, task.n)))
    if max_attempts < 1 or cfg.get('rollin_noise', .012) < 0:
        raise ValueError('max_attempts must be positive and rollin_noise must be nonnegative')
    counts = {name: 0 for name in COUNT_FIELDS}
    by_force, by_takeover, by_pair, by_source, by_direction = {}, {}, {}, {}, {}
    saved, batch_index = 0, 0
    causal_check = None

    def bump(field, info):
        counts[field] += 1
        for table, key in ((by_force, info['force_bucket']),
                           (by_takeover, info['takeover_bucket']),
                           (by_pair, info['force_bucket'] + '/' + info['takeover_bucket']),
                           (by_source, info['motion_id']), (by_direction, info['direction_bucket'])):
            table.setdefault(key, {name: 0 for name in COUNT_FIELDS})[field] += 1

    def progress_report():
        report = dict(saved_successful_episodes=saved, clean_episode_attempts=counts['attempted'],
                      clean_episode_success_rate=counts['succeeded_after_takeover'] / max(1, counts['attempted']),
                      takeover_success_rate=counts['succeeded_after_takeover'] / max(1, counts['takeovers']),
                      rollin_resets=0, rollin_method=method, counts=counts,
                      attempt_definition='One physical reset through termination, including failures before teacher takeover',
                      by_force=by_force, by_takeover=by_takeover,
                      by_force_takeover=by_pair, by_source=by_source,
                      by_direction=by_direction, fixed_attempt_budget=fixed_attempts or None,
                      causal_label_check=causal_check, feature_version=feature_version,
                      teacher_identity=cfg.get('teacher_identity',cfg.get('checkpoint')),
                      teacher_provenance=cfg.get('teacher_provenance'),
                      stop_reference_version=cfg.get('stop_reference_version'),
                      motion_bank_sha256=cfg.get('motion_bank_sha256'),
                      motion_metadata_sha256=cfg.get('motion_metadata_sha256'))
        (out / 'progress.json').write_text(json.dumps(report, indent=2))
        print('V4_RECOVERY_COLLECT', json.dumps(dict(saved=saved, counts=counts)), flush=True)

    while saved < desired and counts['attempted'] < max_attempts:
        task.current_bw = cfg.get('max_bw', .75)
        x = task.reset()
        all_ids = torch.arange(task.n, device=task.device)
        reset_controller(policy, all_ids=all_ids)
        if rollin_policy is not None:
            reset_controller(rollin_policy, all_ids=all_ids)
        causal_check = check_causal_features(task, policy)
        initial_progress = task.env.progress_buf.clone().cpu()
        motion_ids = task.env.motion_manager.motion_ids.clone().cpu()
        strength = task.strength.clone().cpu()
        direction = task.direction.clone().cpu()
        push_start = task.start_step.clone().cpu()
        bucket_ids = (torch.arange(task.n) + batch_index * task.n) % len(schedule)
        batch_index += 1
        offsets = torch.tensor([int(schedule[int(k)]['after_push_end_steps']) for k in bucket_ids])
        takeover_target = push_start + pulse_steps + offsets
        if cfg.get('takeover_at_episode_start', False):
            takeover_target = initial_progress.clone()
            offsets = initial_progress - push_start - pulse_steps
        planned = []
        for j in range(task.n):
            info = dict(motion_id=task.rows[int(motion_ids[j])]['motion_id'],
                        force_bucket=force_bucket(float(strength[j])),
                        takeover_bucket=schedule[int(bucket_ids[j])]['name'],
                        direction_bucket=str(round(float(torch.atan2(direction[j, 1], direction[j, 0])) * 2 / math.pi) % 4))
            planned.append(info)
            bump('attempted', info)
        live = torch.ones(task.n, dtype=torch.bool, device=task.device)
        taken = torch.zeros_like(live)
        target_device = takeover_target.to(task.device)
        prefix, record_start, record_index = {}, {}, {}
        completion = {}
        features, actions, totals, bases = [], [], [], []
        corrections, prior_totals, prior_deltas = [], [], []
        max_steps = int(task.deadline.max()) - int(initial_progress.min()) + 1
        for step in range(max_steps):
            if not torch.equal(x[live], task.history.flatten(1)[live]):
                raise ValueError('Policy input is not the current task observation history')
            newly_taken = live & ~taken & (task.env.progress_buf >= target_device)
            for j in newly_taken.nonzero(as_tuple=True)[0].tolist():
                prefix[j] = task.history[j, :-1].detach().cpu().to(storage_dtype).clone()
                record_start[j] = int(task.env.progress_buf[j])
                record_index[j] = step
                bump('takeovers', planned[j])
            taken |= newly_taken
            features.append(task.history[:, -1].detach().cpu().to(storage_dtype).clone())
            # No strength, direction, push time or takeover flag is passed to
            # either network. The clean label is queried after the real roll-in.
            clean = policy(x)[0].mean
            if not torch.isfinite(clean[live]).all():
                raise ValueError('Nonfinite clean teacher action in a live environment')
            waiting = live & ~taken
            if waiting.any():
                if method in ('bc', 'diffusion'):
                    rollin = rollin_policy(x)[0].mean
                elif method == 'noisyteacher':
                    rollin = clean + torch.randn_like(clean) * cfg.get('rollin_noise', .012)
                else:
                    rollin = clean
                executed_input = torch.where(waiting[:, None], rollin, clean)
            else:
                executed_input = clean
            executed_input = torch.where(live[:, None], executed_input, torch.zeros_like(executed_input))
            x, _, done, info = task.step(executed_input)
            if info['base_normalized'].shape != (task.n, 69):
                raise ValueError('Unexpected base action dimension')
            if not torch.allclose((info['executed_total'] - info['base_normalized'])[live],
                                  info['executed_residual'][live], atol=1e-6, rtol=0):
                raise ValueError('Stored residual differs from actual executed correction')
            for destination, key in ((actions, 'executed_residual'), (totals, 'executed_total'),
                                     (bases, 'base_normalized')):
                destination.append(info[key].detach().cpu().float().clone())
            if not torch.allclose(info['executed_correction'],info['executed_total']-info['prior_total'],atol=2e-6,rtol=0):
                raise ValueError('Warm correction label does not equal the actual saturated action difference')
            for destination,key in ((corrections,'executed_correction'),(prior_totals,'prior_total'),(prior_deltas,'prior_delta')):
                destination.append(info[key].detach().cpu().float().clone())
            for j in (done & live).nonzero(as_tuple=True)[0].tolist():
                success = bool(info['success'][j])
                completion[j] = dict(end_index=step + 1, success=success,
                                     fallen=bool(task.fallen[j]), violation=bool(task.violation[j]))
                if success and (completion[j]['fallen'] or completion[j]['violation']):
                    raise ValueError('Task reported success despite a fall or forbidden contact')
                bump('finished', planned[j])
                if j in record_index:
                    bump('succeeded_after_takeover' if success else 'failed_after_takeover', planned[j])
                else:
                    bump('completed_before_takeover' if success else 'failed_before_takeover', planned[j])
            finished_ids = (done & live).nonzero(as_tuple=True)[0]
            if len(finished_ids):
                reset_controller(policy, ids=finished_ids)
                if rollin_policy is not None:
                    reset_controller(rollin_policy, ids=finished_ids)
            live &= ~done
            if not live.any():
                break
        for j in live.nonzero(as_tuple=True)[0].tolist():
            bump('unresolved', planned[j])
        feature_tensor, action_tensor = torch.stack(features), torch.stack(actions)
        total_tensor, base_tensor = torch.stack(totals), torch.stack(bases)
        correction_tensor,prior_total_tensor,prior_delta_tensor=map(torch.stack,(corrections,prior_totals,prior_deltas))
        for j, finished in sorted(completion.items()):
            if not finished['success'] or j not in record_index or saved >= desired:
                continue
            first, end = record_index[j], finished['end_index']
            if end <= first:
                raise ValueError('A successful teacher continuation must contain executed actions')
            payload = dict(
                feature=feature_tensor[first:end, j].clone(), history_prefix=prefix[j],
                executed_residual=action_tensor[first:end, j].clone(),
                executed_total=total_tensor[first:end, j].clone(),
                base_normalized=base_tensor[first:end, j].clone(),
                executed_correction=correction_tensor[first:end,j].clone(),
                prior_total=prior_total_tensor[first:end,j].clone(),
                prior_delta=prior_delta_tensor[first:end,j].clone(),
                action_target_version='executed_total_minus_composed_terrain_prior_v1',
                teacher_checkpoint_sha256=checkpoint_sha256, frozen_prior=frozen_prior,
                success=True, fallen=finished['fallen'], forbidden_violation=finished['violation'],
                source=task.rows[int(motion_ids[j])], bw=float(strength[j]),
                push_direction=direction[j].clone(), push_start_step=int(push_start[j]),
                record_start_control_step=record_start[j],
                rollin_steps=record_start[j] - int(initial_progress[j]),
                rollin_method=method, rollin_student_checkpoint=cfg.get('rollin_student_checkpoint'),
                teacher_checkpoint=cfg.get('checkpoint'),
                teacher_identity=cfg.get('teacher_identity', cfg.get('checkpoint')),
                teacher_kind=cfg.get('teacher_kind', 'residual'),
                teacher_provenance=cfg.get('teacher_provenance'),
                stop_reference_version=cfg.get('stop_reference_version'),
                motion_bank_sha256=cfg.get('motion_bank_sha256'),
                motion_metadata_sha256=cfg.get('motion_metadata_sha256'),
                takeover_bucket=planned[j]['takeover_bucket'],
                takeover_after_push_end_steps=int(offsets[j]),
                planned_takeover_control_step=int(takeover_target[j]),
                force_bucket=planned[j]['force_bucket'], feature_version=feature_version,
                causal_label_check=causal_check,
                label='Continuous clean teacher execution after actual current-state takeover; normalized executed action minus tracker action',
            )
            torch.save(payload, out / f'episode_{saved:05d}.pt')
            saved += 1
            bump('saved', planned[j])
        progress_report()
    if saved < desired and not fixed_attempts:
        raise RuntimeError(f'Recovery collection exhausted {counts["attempted"]} attempts with {saved}/{desired} successful continuations; see progress.json')
    if saved == 0:
        raise RuntimeError('The fixed collection budget produced no successful teacher continuations; see progress.json')
    print('V4_RECOVERY_COLLECT_COMPLETE', saved, flush=True)
    return dict(saved=saved, counts=counts)


collect_recovery = collect
