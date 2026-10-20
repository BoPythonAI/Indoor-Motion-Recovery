"""Optional training-only late push schedule; no policy feature additions.

Call reset_after AFTER the original task reset. Original nominal decisions,
strengths, direction samples and early schedule are retained where applicable.
Call observe_step(info, gaps, done) after each original physics step. Diagnostics
do not enter task.history or observations. All probabilities below are fixed.
"""
import torch


VERSION = 'surface_late_push_v1'
GROUPS = ('nominal', 'original_random', 'late_random', 'late_toward')
COUNTS = ('started', 'steps', 'near10_steps', 'near05_steps', 'forbidden_steps',
          'completed', 'near10_episodes', 'near05_episodes', 'forbidden_episodes',
          'direction_fallbacks', 'abandoned', 'observed_push_starts')


class SurfaceCurriculum:
    def __init__(self, task, cfg):
        option = cfg.get('surface_curriculum', False)
        requested = option.get('enabled', False) if isinstance(option, dict) else bool(option)
        self.enabled = bool(cfg.get('mode') == 'train' and requested)
        self.task, self.cfg = task, cfg
        self.device, self.n = task.device, task.n
        if self.enabled and any(row.get('split') != 'train' for row in task.rows):
            raise ValueError('Surface curriculum only accepts the existing training split')
        self.group = torch.full((self.n,), -1, device=self.device, dtype=torch.long)
        self.active = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        self.episode_steps = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.near10 = torch.zeros_like(self.active)
        self.near05 = torch.zeros_like(self.active)
        self.forbidden = torch.zeros_like(self.active)
        self.push_seen = torch.zeros_like(self.active)
        self.counts = torch.zeros((len(GROUPS), len(COUNTS)), device=self.device, dtype=torch.long)
        self.first_push_phase_sum = torch.zeros(len(GROUPS), device=self.device, dtype=torch.float64)

    def _ids(self, ids):
        if ids is None:
            return self.task.ids
        return torch.as_tensor(ids, device=self.device, dtype=torch.long).flatten()

    def _add(self, column, groups):
        self.counts[:, COUNTS.index(column)] += torch.bincount(groups, minlength=len(GROUPS))

    @torch.no_grad()
    def reset_after(self, ids=None, cases=None):
        if not self.enabled or self.cfg.get('mode') != 'train':
            return None  # In particular: no RNG consumption in evaluate/collect/default-off.
        ids = self._ids(ids)
        if cases is not None:
            self.active[ids] = False
            self.group[ids] = -1
            return None  # Explicit case schedule is always owned by the original task.
        if not len(ids):
            return None
        task = self.task
        strength = task.strength[ids]
        nonzero = strength > 0
        legal = (~nonzero) | torch.isclose(strength, strength.new_tensor(.45), atol=1e-6, rtol=0) | torch.isclose(
            strength, strength.new_tensor(.75), atol=1e-6, rtol=0)
        if not bool(legal.all()):
            raise ValueError('This curriculum preserves the original 0/.45/.75 bodyweight levels')
        # Keep original 20% Bernoulli nominal choices exactly. Among nonzero
        # cases: 50% untouched original, 25% late random, 25% late toward.
        groups = torch.zeros(len(ids), device=self.device, dtype=torch.long)
        nonzero_ids = nonzero.nonzero(as_tuple=True)[0]
        choice = torch.rand(len(nonzero_ids), device=self.device)
        groups[nonzero_ids] = torch.where(choice < .5, 1, torch.where(choice < .75, 2, 3))
        late_local = (groups >= 2).nonzero(as_tuple=True)[0]
        late_ids = ids[late_local]
        if len(late_ids):
            mids = task.env.motion_manager.motion_ids[late_ids]
            late_phase = .85 + .20 * torch.rand(len(late_ids), device=self.device)
            late_start = (task.source_frames[mids] * late_phase).long().clamp_min(8)
            # Existing eligibility is progress > start+36 and requires 30 stable
            # steps. Leave that exact criterion/deadline unchanged, but reject a
            # schedule that could never complete within it.
            if bool((late_start + 66 > task.deadline[late_ids]).any()):
                raise ValueError('Late push leaves fewer than the original 66 steps needed before deadline')
        else:
            late_start = task.start_step[late_ids].clone()

        toward_ids = ids[(groups == 3).nonzero(as_tuple=True)[0]]
        fallback_ids = toward_ids[:0]
        valid_toward = toward_ids
        toward_xy = task.direction[toward_ids, :2].clone()
        if len(toward_ids):
            goal = task.goal[toward_ids, :2]
            lower = task.box_center[toward_ids, :2] - task.box_half[toward_ids, :2]
            upper = task.box_center[toward_ids, :2] + task.box_half[toward_ids, :2]
            delta = goal.clamp(min=lower, max=upper) - goal
            distance = delta.norm(dim=-1)
            if not bool(torch.isfinite(delta).all()):
                raise ValueError('Nonfinite world-space goal or forbidden AABB')
            valid = distance > 1e-6
            valid_toward, fallback_ids = toward_ids[valid], toward_ids[~valid]
            toward_xy = delta[valid] / distance[valid, None]
            # Inside the horizontal AABB there is no unique approach direction.
            # Preserve the already sampled random direction; never create zero force.

        abandoned = ids[self.active[ids] & (self.episode_steps[ids] > 0)]
        self._add('abandoned', self.group[abandoned])
        # All validation finished before the only writes to task control state.
        task.start_step[late_ids] = late_start
        task.direction[valid_toward, :2] = toward_xy
        task.direction[valid_toward, 2] = 0
        self.group[ids] = groups
        self.active[ids] = True
        self.episode_steps[ids] = 0
        self.near10[ids] = False; self.near05[ids] = False
        self.forbidden[ids] = False; self.push_seen[ids] = False
        self._add('started', groups)
        self._add('direction_fallbacks', self.group[fallback_ids])
        return groups.detach().clone()  # Diagnostic only; never append to features.

    @torch.no_grad()
    def observe_step(self, info, foot_clearance, done):
        if not self.enabled or self.cfg.get('mode') != 'train':
            return
        if foot_clearance.shape not in ((self.n,), (self.n, 4)) or done.shape != (self.n,):
            raise ValueError('Expected foot clearance [N,4] or [N], and original done [N]')
        minimum = foot_clearance.amin(-1) if foot_clearance.ndim == 2 else foot_clearance
        ids = self.active.nonzero(as_tuple=True)[0]
        groups = self.group[ids]
        if not bool(torch.isfinite(minimum[ids]).all()):
            raise ValueError('Nonfinite foot surface gap in training diagnostics')
        hit10, hit05 = minimum[ids] < .10, minimum[ids] < .05
        violation = info['violation'][ids].bool()
        self._add('steps', groups)
        self._add('near10_steps', groups[hit10]); self._add('near05_steps', groups[hit05])
        self._add('forbidden_steps', groups[violation])
        self.episode_steps[ids] += 1
        self.near10[ids] |= hit10; self.near05[ids] |= hit05
        self.forbidden[ids] |= violation
        if 'external_force' in info:
            applying = info['external_force'][ids].norm(dim=-1) > 0
            first = ids[applying & ~self.push_seen[ids]]
            first_groups = self.group[first]
            self._add('observed_push_starts', first_groups)
            mids = self.task.env.motion_manager.motion_ids[first]
            end_time = (self.task.source_frames[mids] - 1) * self.task.env.motion_lib.motion_dt[mids]
            phase = self.task.env.motion_manager.motion_times[first] / end_time.clamp_min(1e-6)
            self.first_push_phase_sum.scatter_add_(0, first_groups, phase.to(torch.float64))
            self.push_seen[first] = True
        finished = ids[done[ids].bool()]
        finished_groups = self.group[finished]
        self._add('completed', finished_groups)
        self._add('near10_episodes', finished_groups[self.near10[finished]])
        self._add('near05_episodes', finished_groups[self.near05[finished]])
        self._add('forbidden_episodes', finished_groups[self.forbidden[finished]])
        self.active[finished] = False  # Subsequent padded steps are not duplicate episodes.

    @torch.no_grad()
    def metrics(self, reset=False):
        rows = self.counts.detach().cpu().tolist()
        phase_sums = self.first_push_phase_sum.detach().cpu().tolist()
        report = dict(version=VERSION, enabled=bool(self.enabled and self.cfg.get('mode') == 'train'),
                      gap_measure='minimum post-step foot gap supplied by the surface task', groups={})
        for index, name in enumerate(GROUPS):
            row = dict(zip(COUNTS, rows[index]))
            for prefix in ('near10', 'near05', 'forbidden'):
                row[prefix + '_step_rate'] = row[prefix + '_steps'] / row['steps'] if row['steps'] else None
                row[prefix + '_episode_rate'] = row[prefix + '_episodes'] / row['completed'] if row['completed'] else None
            count = row['observed_push_starts']
            row['mean_first_push_post_reference_phase'] = phase_sums[index] / count if count else None
            report['groups'][name] = row
        if reset:
            self.counts.zero_(); self.first_push_phase_sum.zero_()
            # Preserve open episodes: full-episode flags are charged in the window
            # where that episode ends; started/completed can differ within a window.
        return report
