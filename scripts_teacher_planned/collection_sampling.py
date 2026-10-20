"""Balanced collection attempts, isolated from policy inputs and PPO resets."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


class BalancedCases:
    def __init__(self, rows, seed, strengths=(0., .45, .75, .45, .75),
                 directions=(0, 1, 2, 3), phases=(.3, .6)):
        if not rows or any(row['split'] != 'train' for row in rows):
            raise ValueError('Balanced collection requires only training sources')
        self.rows = rows
        self.families = defaultdict(list)
        for i, row in enumerate(rows):
            self.families[row['family']].append(i)
        self.families = dict(sorted(self.families.items()))
        self.rng = random.Random(seed)
        self.strengths, self.directions, self.phases = list(strengths), list(directions), list(phases)
        if not self.strengths or not self.directions or not self.phases:
            raise ValueError('All balance axes need values')
        self.block_size = len(self.families) * len(self.directions) * len(self.phases)
        self.cycle_size = self.block_size * len(self.strengths)
        self.block = 0
        self.pending = []
        self.placement_cursor = Counter()
        self.counts = {axis: Counter() for axis in ('source', 'family', 'strength', 'direction', 'phase')}

    def next_cases(self, n):
        result = []
        while len(result) < n:
            if not self.pending:
                strength = self.strengths[self.block % len(self.strengths)]
                block = []
                for family, mids in self.families.items():
                    # Diagonal placement assignment avoids binding each of the
                    # actual two sibling placements to a fixed push phase.
                    # Repeated strengths rotate the assignment independently.
                    key = (family, float(strength))
                    offset = self.placement_cursor[key]
                    for d_index, direction in enumerate(self.directions):
                        for p_index, phase in enumerate(self.phases):
                            mid = mids[(d_index + p_index + offset) % len(mids)]
                            block.append(dict(motion=mid, bw=float(strength),
                                              direction=direction, phase=phase))
                    self.placement_cursor[key] += 1
                self.rng.shuffle(block)
                self.pending = block
                self.block += 1
            take = min(n - len(result), len(self.pending))
            result.extend(self.pending[:take])
            self.pending = self.pending[take:]
        for case in result:
            row = self.rows[case['motion']]
            for axis, value in (('source', row['motion_id']), ('family', row['family']),
                                ('strength', case['bw']), ('direction', case['direction']), ('phase', case['phase'])):
                self.counts[axis][str(value)] += 1
        return result

    def report(self):
        return dict(kind='Balanced physical attempts; successful outcomes are not guaranteed balanced',
                    families=len(self.families), sources=len(self.rows),
                    block_size=self.block_size, cycle_size=self.cycle_size,
                    strengths=self.strengths, directions=self.directions, phases=self.phases,
                    counts={name: dict(values) for name, values in self.counts.items()},
                    teacher_inputs='Unchanged current-history tensors; reset case metadata stays in the collector')


def run_collection(task, policy, cfg):
    if cfg.get('feature_version', task.feature_version) != task.feature_version:
        raise ValueError('Collection config and task feature conventions differ')
    kind = cfg.get('collector', 'full')
    if kind == 'full':
        from collect import collect
    elif kind == 'recovery':
        from collect_recovery import collect
    else:
        raise ValueError('collector must be full or recovery')
    if not cfg.get('balanced_collection', False):
        return collect(task, policy, cfg)
    sampler = BalancedCases(task.rows, cfg.get('collection_seed', cfg.get('seed', 17)),
                            cfg.get('collection_strength_schedule', [0., .45, .75, .45, .75]),
                            cfg.get('collection_directions', [0, 1, 2, 3]),
                            cfg.get('collection_phases', [.3, .6]))
    fixed_attempts = int(cfg.get('fixed_attempts', 0))
    if not fixed_attempts or fixed_attempts % sampler.cycle_size:
        raise ValueError('Balanced pilot needs fixed_attempts equal to complete sampling cycles')
    original_reset = task.reset

    def reset(ids=None, cases=None):
        if ids is not None or cases is not None:
            raise ValueError('Balanced collection only resets whole batches; partial resets would alter quotas')
        return original_reset(cases=sampler.next_cases(task.n))

    task.reset = reset
    try:
        return collect(task, policy, cfg)
    finally:
        task.reset = original_reset
        output = Path(cfg['output'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'sampling_report.json').write_text(json.dumps(sampler.report(), indent=2))
