"""Offline BC/diffusion correction distillation; run only after teacher qualification."""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from contracts import POLICY_KIND
from dataset import load_episodes, correction_statistics
from model import ChunkPolicy, update_ema


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, files = load_episodes(args.data, device, args.nominal_only)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Student training requires a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    weights, sampling = None, dict(order='uniform_frame')
    if args.sampling == 'balanced':
        from dataset_sampling import balanced_frame_weights
        weights, sampling = balanced_frame_weights(dataset.sampling)
        weights = weights.to(device)
        if len(weights) != len(dataset.valid_indices):
            raise ValueError('Sampling weights do not align with supervised action frames')
    model = ChunkPolicy(dataset.features.shape[-1] * 4, args.kind, args.horizon).to(device)
    valid_actions = dataset.actions[dataset.valid_indices]
    correction_stats = correction_statistics(valid_actions)
    model.action_mean.copy_(valid_actions.mean(0))
    model.action_std.copy_(valid_actions.std(0, unbiased=False).clamp_min(.005))
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    manifest = dict(**dataset.contract, policy_kind=POLICY_KIND, files=files,
        directories=args.data, successful_episodes=dataset.episode_count,
        supervised_frames=len(dataset.valid_indices), frames_including_history=len(dataset.features),
        seed=args.seed, steps=args.steps, kind=args.kind, horizon=args.horizon,
        correction_statistics=correction_stats,
        sampling=sampling, default_weights='ema', ema_max_decay=args.ema_decay, ema_power=args.ema_power,
        label='Clean executed correction after real-state takeover, with a same-state composed frozen terrain prior')
    (out / 'dataset.json').write_text(json.dumps(manifest, indent=2))
    (out / 'correction_statistics.json').write_text(json.dumps(correction_stats, indent=2))
    print('WARM_CORRECTION_SCALE', json.dumps({key: correction_stats[key] for key in
        ('frames', 'correction_rms', 'correction_centered_rms', 'std_min', 'std_median', 'std_max',
         'normalization_floor', 'dimensions_below_floor', 'normalized_centered_rms')}), flush=True)

    def save(path, step):
        state = dict(**dataset.contract, policy_kind=POLICY_KIND,
            model=ema.state_dict(), ema_model=ema.state_dict(), online_model=model.state_dict(),
            kind=model.kind, obs_dim=model.obs_dim, horizon=model.horizon, steps=step,
            seed=args.seed, episodes=dataset.episode_count, supervised_frames=len(dataset.valid_indices),
            default_weights='ema', ema_max_decay=args.ema_decay, ema_power=args.ema_power,
            nominal_only=args.nominal_only, implementation='student_surface_correction_v1')
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary); temporary.replace(path)

    started = time.perf_counter()
    for step in range(args.steps):
        chosen = (torch.multinomial(weights, args.batch_size, replacement=True) if weights is not None
                  else torch.randint(len(dataset.valid_indices), (args.batch_size,), device=device))
        history, target, mask = dataset.chunk(dataset.valid_indices[chosen], args.horizon)
        loss = model.loss(history, target, mask)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        decay = update_ema(ema, model, step, args.ema_decay, args.ema_power)
        if step % 100 == 0:
            report = dict(step=step, loss=float(loss.detach()), seconds=time.perf_counter() - started, ema_decay=decay)
            print(json.dumps(report), flush=True)
            with (out / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(report) + '\n')
        if (step + 1) % 5000 == 0:
            save(out / f'step_{step + 1:05d}.pt', step + 1)
    save(out / 'last.pt', args.steps)
    print('WARM_STUDENT_TRAIN_COMPLETE', str(out / 'last.pt'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--kind', choices=['bc', 'diffusion'], required=True)
    parser.add_argument('--steps', type=int, default=15000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--nominal-only', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--ema-decay', type=float, default=.9999)
    parser.add_argument('--ema-power', type=float, default=.75)
    parser.add_argument('--device')
    parser.add_argument('--sampling', choices=['uniform', 'balanced'], default='uniform')
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.horizon) < 1 or not 0 <= args.ema_decay < 1 or args.ema_power <= 0:
        parser.error('Positive steps, batch size, horizon, EMA power and 0 <= EMA decay < 1 required')
    train(args)
