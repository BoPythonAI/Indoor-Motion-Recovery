"""Fixed terrain baseline represented in the existing flat residual actuator API."""
from dataclasses import dataclass
import torch

from official_tracker import load_official_actor, check_runtime_contract
from motion_reference import file_sha256


@dataclass
class PriorStep:
    delta: torch.Tensor
    flat_raw: torch.Tensor
    terrain_raw: torch.Tensor


class FrozenTerrainPrior:
    def __init__(self, agent, config):
        self.agent, self.env = agent, agent.env
        self.flat_actor = agent.model._actor
        self.flat_actor.eval().requires_grad_(False)
        self.terrain_actor, saved = load_official_actor(config['terrain_checkpoint'], self.env.device)
        self.contract_report = check_runtime_contract(self.env, saved)
        if 'terrain' in self.flat_actor.mu.config.in_keys or 'terrain' not in self.terrain_actor.mu.config.in_keys:
            raise ValueError('Warm prior requires the FLAT base environment and official terrain-aware prior')
        self.points = saved['terrain'].num_samples_per_axis ** 2
        if self.points != 256 or self.terrain_actor.config.num_out != self.env.robot_config.number_of_actions:
            raise ValueError('Warm prior expects the validated terrain256 / common actuator contract')
        if self.env.config.action_config['action_transform'] != 'tanh':
            raise ValueError('Warm prior requires existing normalized tanh actuator composition')
        self.identity = dict(kind='official_terrain_prior',
            flat_checkpoint=config['tracker_checkpoint'], flat_sha256=file_sha256(config['tracker_checkpoint']),
            terrain_checkpoint=config['terrain_checkpoint'], terrain_sha256=file_sha256(config['terrain_checkpoint']))

    def _mean(self, actor, obs):
        td = self.agent.obs_dict_to_tensordict(self.agent.add_agent_info_to_obs(dict(obs)))
        return actor.mu(td)[actor.config.mu_key]

    @torch.no_grad()
    def current(self):
        obs = self.env.get_obs()
        if (not getattr(self.env.terrain, '_scene_heightmap_installed', False)
                or 'get_ground_heights' not in self.env.terrain.__dict__):
            raise RuntimeError('Install the selected V5 scene floor and scene height-map adapter')
        if obs['terrain'].shape != (self.env.num_envs, 256) or not torch.isfinite(obs['terrain']).all():
            raise ValueError('Current terrain observation is missing or invalid')
        flat = self._mean(self.flat_actor, obs)
        terrain = self._mean(self.terrain_actor, obs)
        return PriorStep(delta=terrain.tanh()-flat.tanh(), flat_raw=flat, terrain_raw=terrain)
