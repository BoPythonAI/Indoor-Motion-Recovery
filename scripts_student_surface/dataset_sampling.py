"""Metadata-only sampling balance; no perturbation labels enter model inputs."""
from collections import defaultdict
import math
import torch


def strength_group(bw):
    if math.isclose(bw, 0., abs_tol=1e-5):
        return 'nominal'
    return '0_to_0.45' if bw <= .45001 else '0.45_to_0.75'


def balanced_frame_weights(episodes):
    # Equal streams -> equal families -> 20/40/40 force strata -> equal
    # directions -> equal episodes -> uniform frames within each episode.
    tree = {}
    for i, episode in enumerate(episodes):
        key = (episode['stream'], episode['family'], strength_group(episode['bw']), episode['direction'])
        stream, family, strength, direction = key
        tree.setdefault(stream, {}).setdefault(family, {}).setdefault(strength, {}).setdefault(direction, []).append(i)
    desired_force = {'nominal': .2, '0_to_0.45': .4, '0.45_to_0.75': .4}
    masses = torch.zeros(len(episodes), dtype=torch.float64)
    for stream, families in tree.items():
        for family, strengths in families.items():
            total_strength = sum(desired_force[s] for s in strengths)
            for strength, directions in strengths.items():
                mass = (1 / len(tree) / len(families) * desired_force[strength] / total_strength / len(directions))
                for indices in directions.values():
                    masses[indices] = mass / len(indices)
    weights = torch.cat([torch.full((episode['frames'],), masses[i] / episode['frames'], dtype=torch.float64)
                         for i, episode in enumerate(episodes)])
    if not torch.isfinite(weights).all() or not torch.isclose(weights.sum(), torch.tensor(1., dtype=torch.float64)):
        raise ValueError('Invalid balanced frame probability mass')
    summary = {axis: defaultdict(float) for axis in ('stream', 'family', 'strength', 'direction')}
    for i, episode in enumerate(episodes):
        values = dict(stream=episode['stream'], family=episode['family'],
                      strength=strength_group(episode['bw']), direction=episode['direction'])
        for axis, value in values.items():
            summary[axis][str(value)] += float(masses[i])
    return weights.float(), dict(
        order='equal_stream/equal_family/force_20_40_40/equal_direction/equal_episode/uniform_frame',
        missing_strata='Renormalize only among available successful strata; never synthesize missing recovery labels',
        input_leakage='Metadata changes sample selection only, never observation features',
        probability_mass={axis: dict(values) for axis, values in summary.items()})
