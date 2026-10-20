"""Warm task plus current foot-box clearance; unchanged physics and success tests."""
from pathlib import Path
import torch

from warm_task_base import WarmRecoveryTask as OriginalWarmTask
from foot_clearance import FootClearance
from motion_reference import file_sha256
from surface_contract import FEATURE_VERSION, GEOMETRY_VERSION, PROXIMITY_REWARD


def replace_proximity_penalty(reward, bodypoint_gap, surface_gap):
    """Replace only the inherited 2*exp(-gap/.12) body-point proximity cost."""
    return reward + 2*torch.exp(-bodypoint_gap/.12) - 2*torch.exp(-surface_gap/.12)


class WarmRecoveryTask(OriginalWarmTask):
    feature_version = FEATURE_VERSION

    def __init__(self, agent, config):
        super().__init__(agent, config)
        if not self.scene_dir:
            raise ValueError('This surface branch requires the unchanged real-scene task')
        from protomotions.assets import resolve_asset_root
        asset=self.env.robot_config.asset
        path=(Path(resolve_asset_root(asset.asset_root))/asset.asset_file_name).resolve()
        actual=file_sha256(path)
        if config.get('geometry_version',GEOMETRY_VERSION)!=GEOMETRY_VERSION:
            raise ValueError('Surface geometry version mismatch')
        if config.get('surface_mjcf_sha256',actual)!=actual:
            raise ValueError('Runtime collision MJCF differs from the migrated/trained surface teacher')
        config.update(geometry_version=GEOMETRY_VERSION,surface_mjcf_path=str(path),
                      surface_mjcf_sha256=actual,proximity_reward=PROXIMITY_REWARD)
        self.foot_geometry=FootClearance(path,self.names).to(self.device)
        self._surface_result=self._surface_features=self._surface_progress=None
        self._bodypoint_gap=self._surface_gap=None
        from surface_curriculum import SurfaceCurriculum
        self.surface_curriculum=SurfaceCurriculum(self,config)

    def _invalidate_surface(self):
        self._surface_result=self._surface_features=self._surface_progress=None
        self._bodypoint_gap=self._surface_gap=None

    @torch.no_grad()
    def _sample_surface(self):
        if self._surface_result is not None and torch.equal(self._surface_progress,self.env.progress_buf):
            return
        state=self.env.simulator.get_robot_state()
        result=self.foot_geometry(state.rigid_body_pos,state.rigid_body_rot,self.box_center,self.box_half)
        outside=(state.rigid_body_pos-self.box_center[:,None]).abs()-self.box_half[:,None]
        body_distances=outside.clamp_min(0).norm(dim=-1)
        self._surface_result=result
        self._surface_features=self.foot_geometry.features(result,state.rigid_body_rot[:,0])
        self._bodypoint_gap=body_distances.amin(-1)
        self._surface_gap=self.foot_geometry.proximity(body_distances,result)
        self._surface_progress=self.env.progress_buf.clone()

    @torch.no_grad()
    def reset(self, ids=None, cases=None):
        self._invalidate_surface()
        history=super().reset(ids,cases)
        self.surface_curriculum.reset_after(ids,cases)
        return history

    @torch.no_grad()
    def features(self):
        original=super().features()
        self._sample_surface()
        return torch.cat([original,self._surface_features],-1)

    @torch.no_grad()
    def step(self, full_residual_sample):
        self._invalidate_surface()
        history,reward,done,info=super().step(full_residual_sample)
        # V5.step called self.features() after this exact physics step. Reuse its one sample.
        if self._surface_result is None:
            raise RuntimeError('Expected the post-step surface sample from feature construction')
        reward=replace_proximity_penalty(reward,self._bodypoint_gap,self._surface_gap)
        info.update(foot_sat_gap=self._surface_result.gaps,bodypoint_proximity_gap=self._bodypoint_gap,
                    surface_proximity_gap=self._surface_gap)
        self.surface_curriculum.observe_step(info,self._surface_result.gaps,done)
        return history,reward,done,info
