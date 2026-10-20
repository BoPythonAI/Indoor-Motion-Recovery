"""Causal takeover histories and terminal-masked chunks of executed corrections."""
import math
from pathlib import Path
import torch
from contracts import provenance_from_episode, signature


def correction_statistics(actions, normalization_floor=.005):
    """Report the actual small target scale before applying the inherited floor."""
    mean, std = actions.mean(0), actions.std(0, unbiased=False)
    scale = std.clamp_min(normalization_floor)
    centered = actions - mean
    normalized_std = std / scale
    normalized_abs = (centered / scale).abs()
    outside_diffusion_clip = normalized_abs > 8
    below = std < normalization_floor
    return dict(frames=len(actions), action_dimensions=69,
        correction_rms=float(actions.square().mean().sqrt()),
        correction_mean_abs=float(actions.abs().mean()),
        correction_centered_rms=float(centered.square().mean().sqrt()),
        correction_abs_max=float(actions.abs().max()),
        per_dimension_mean=mean.cpu().tolist(), per_dimension_rms=actions.square().mean(0).sqrt().cpu().tolist(),
        per_dimension_std=std.cpu().tolist(), std_min=float(std.min()), std_median=float(std.median()), std_max=float(std.max()),
        normalization_floor=normalization_floor, dimensions_below_floor=int(below.sum()),
        fraction_dimensions_below_floor=float(below.float().mean()),
        per_dimension_normalization_scale=scale.cpu().tolist(),
        per_dimension_normalized_target_std=normalized_std.cpu().tolist(),
        normalized_centered_rms=float((centered / scale).square().mean().sqrt()),
        normalized_target_abs_max=float(normalized_abs.max()),
        normalized_target_abs_max_per_dimension=normalized_abs.amax(0).cpu().tolist(),
        normalized_target_abs_gt8_element_fraction=float(outside_diffusion_clip.float().mean()),
        normalized_target_abs_gt8_frame_fraction=float(outside_diffusion_clip.any(-1).float().mean()),
        normalized_target_abs_gt8_fraction_per_dimension=outside_diffusion_clip.float().mean(0).cpu().tolist(),
        diffusion_clean_clip=8.,
        interpretation='The fixed std floor is inherited unchanged. A dimension below it has normalized std < 1; these statistics expose that reduction without altering labels.')


def validate_episode(data):
    contract = provenance_from_episode(data)
    if data['source']['split'] != 'train':
        raise ValueError('Student labels may only come from the training source split')
    if not data.get('success') or data.get('fallen') or data.get('forbidden_violation'):
        raise ValueError('Only successful clean teacher continuations are labels')
    feature = data['feature'].float()
    correction = data['executed_correction']
    total, prior = data['executed_total'], data['prior_total']
    if correction.dtype != torch.float32 or total.dtype != torch.float32 or prior.dtype != torch.float32:
        raise ValueError('Correction and both total actions must be float32; features may be float16')
    if feature.ndim != 2 or len(feature) < 1 or correction.shape != (len(feature), 69):
        raise ValueError('Expected aligned feature[T,D] and executed_correction[T,69]')
    if total.shape != correction.shape or prior.shape != correction.shape:
        raise ValueError('Action/prior labels are not aligned')
    prefix = data['history_prefix'].float()
    if prefix.shape != (3, feature.shape[1]):
        raise ValueError('Takeover needs exactly the preceding three feature frames')
    if not all(torch.isfinite(t).all() for t in (feature, prefix, correction, total, prior)):
        raise ValueError('Nonfinite state or executed action in dataset')
    if not torch.allclose(correction, total - prior, atol=2e-6, rtol=0):
        raise ValueError('Target is not actual executed_total minus same-state composed terrain prior')
    if total.abs().max() > 1.000001 or prior.abs().max() > 1.000001:
        raise ValueError('Labels must be normalized executed actions after action composition')
    return contract, feature, correction, prefix


class Episodes:
    def __init__(self, records, device='cpu'):
        features, actions, starts, ends, valid = [], [], [], [], []
        cursor, identities = 0, set()
        self.sampling, self.contract, self.episode_count = [], None, 0
        for data, stream in records:
            contract, f, a, prefix = validate_episode(data)
            identities.add(signature(contract))
            self.contract = self.contract or contract
            if len(identities) != 1:
                raise ValueError('Do not mix teacher weights, prior weights, motion banks or action contracts')
            direction = int(round(float(torch.atan2(data['push_direction'][1], data['push_direction'][0])) * 2 / math.pi)) % 4
            self.sampling.append(dict(stream=stream, family=data['source']['family'],
                                      bw=float(data['bw']), direction=direction, frames=len(f)))
            f = torch.cat([prefix, f])
            a = torch.cat([torch.zeros(3, 69), a])
            features.append(f); actions.append(a)
            starts.extend([cursor] * len(f)); ends.extend([cursor + len(f)] * len(f))
            valid.extend(range(cursor + 3, cursor + len(f)))
            cursor += len(f); self.episode_count += 1
        if not cursor:
            raise ValueError('No successful warm correction episodes')
        self.features, self.actions = torch.cat(features).to(device), torch.cat(actions).to(device)
        self.start, self.end = torch.tensor(starts, device=device), torch.tensor(ends, device=device)
        self.valid_indices = torch.tensor(valid, device=device)

    def chunk(self, indices, horizon=6):
        if torch.any(indices < self.start[indices] + 3):
            raise ValueError('History prefix rows cannot be supervised action starts')
        history = indices[:, None] + torch.arange(-3, 1, device=indices.device)[None]
        future = indices[:, None] + torch.arange(horizon, device=indices.device)[None]
        mask = future < self.end[indices, None]
        future = torch.minimum(future, self.end[indices, None] - 1)
        return self.features[history].flatten(1), self.actions[future], mask.float()


def load_episodes(directories, device='cpu', nominal_only=False):
    roots = [Path(path).resolve() for path in directories]
    files = sorted({path for root in roots for path in root.rglob('episode_*.pt')})
    if not files:
        raise ValueError('No episode files found')
    records, manifest = [], []
    for path in files:
        streams = [str(root) for root in roots if path.is_relative_to(root)]
        if len(streams) != 1:
            raise ValueError('Dataset directories must not overlap')
        data = torch.load(path, map_location='cpu', weights_only=False)
        # Reject legacy/mixed data before optional nominal filtering.
        validate_episode(data)
        if nominal_only and data['bw'] != 0:
            continue
        records.append((data, streams[0]))
        manifest.append(dict(path=str(path), bytes=path.stat().st_size,
                             motion_id=data['source']['motion_id'], frames=len(data['feature'])))
    return Episodes(records, device), manifest
