"""Online normalized PD residuals with physically executed action history."""
import json
import os
import time
from pathlib import Path
import torch


def compose_action(base_raw, residual):
    """The frozen tracker uses tanh, so residuals live AFTER that transform.

    Invert the transform to use the existing env action path and keep the raw
    action history expected by the pretrained tracker. Zero residual is exact.
    """
    base = torch.tanh(base_raw)
    total = (base + residual).clamp(-1 + 1e-6, 1 - 1e-6)
    raw = torch.atanh(total)
    raw = torch.where(residual == 0, base_raw, raw)
    return raw, torch.tanh(raw), torch.tanh(raw) - base


def base_action(agent, obs):
    td = agent.obs_dict_to_tensordict(agent.add_agent_info_to_obs(obs))
    return agent.model._actor(td)['mean_action']


def run(agent):
    cfg = json.loads(Path(os.environ['V4_CONFIG']).read_text())
    verified=os.environ.get('V4_VERIFIED_MOTION_REFERENCE')
    if not verified:raise ValueError('Use warm entry.py to bind the actual motion bank')
    cfg.update(json.loads(verified))
    torch.manual_seed(cfg.get('seed', 17))
    torch.set_num_threads(4)
    agent.eval()
    agent.evaluator._disable_perturbations()
    env = agent.env
    assert env.robot_config.number_of_actions == 69
    assert env.config.action_config['action_transform'] == 'tanh'
    obs, _ = env.reset()
    root = Path(cfg['output'])
    root.mkdir(parents=True, exist_ok=True)
    print('V4_READY', json.dumps({'envs': env.num_envs, 'dt': env.dt,
          'obs': {k:list(v.shape) for k,v in obs.items()},
          'actions': env.robot_config.number_of_actions, 'body_names':env.robot_config.kinematic_info.body_names}), flush=True)
    if cfg.get('mode', 'smoke') != 'smoke':
        from teacher import train_or_evaluate
        return train_or_evaluate(agent, cfg)
    heights = []
    errors = []
    raw_error = 0.
    history_error = 0.
    start = time.perf_counter()
    with torch.no_grad():
        for k in range(cfg.get('steps', 180)):
            a = base_action(agent, obs)
            raw, total, executed = compose_action(a, torch.zeros_like(a))
            raw_error = max(raw_error, float((a-raw).abs().max()))
            obs, rew, done, term, info = env.step(raw)
            s = env.simulator.get_robot_state()
            heights.append(s.rigid_body_pos[:,0,2].clone())
            errors.append((s.rigid_body_pos-env.context.mimic.ref_state.rigid_body_pos).norm(dim=-1).mean(-1))
            history_error = max(history_error, float((env._current_raw_action-raw).abs().max()))
            if k % 60 == 59:
                print('V4_SMOKE_STEP',k+1, 'height',float(heights[-1].mean()),'error',float(errors[-1].mean()),flush=True)
            if done.any():
                obs, _ = env.reset(done.nonzero(as_tuple=True)[0])
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    h,e = torch.stack(heights), torch.stack(errors)
    report = {'mode':'smoke','envs':env.num_envs,'steps':len(heights),'seconds':elapsed,
              'transitions_per_second':len(heights)*env.num_envs/elapsed,
              'zero_residual_raw_error':raw_error,'executed_history_error':history_error,
              'min_root_height':float(h.min()),'mean_tracking_error':float(e.mean()),
              'finite':bool(torch.isfinite(h).all() and torch.isfinite(e).all()),
              'torch_peak_vram_mib':torch.cuda.max_memory_allocated()/2**20}
    (root/'smoke.json').write_text(json.dumps(report,indent=2))
    print('V4_COMPLETE',json.dumps(report),flush=True)
