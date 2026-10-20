"""Synthetic CPU check of the surface wrapper's sampling/history/reward contract."""
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import torch


class OriginalTask:
    def __init__(self,agent,config):
        self.agent=agent;self.env=agent.env;self.cfg=config
        self.n=2;self.device='cpu';self.names=self.env.names;self.scene_dir='fake'
        self.ids=torch.arange(self.n);self.rows=[dict(split='val')]
        self.box_center=torch.tensor([[.33,0.,0.],[.4,0.,0.]])
        self.box_half=torch.tensor([[.1,.2,.2],[.1,.2,.2]])
        self.history=None
        self.strength=torch.zeros(2);self.start_step=torch.zeros(2);self.direction=torch.zeros(2,3)

    def reset(self,ids=None,cases=None):
        ids=self.ids if ids is None else ids
        self.env.progress_buf[ids]=0
        self.env.state.rigid_body_pos[ids]=0
        feature=self.features()
        if self.history is None:self.history=feature[:,None].repeat(1,4,1)
        else:self.history[ids]=feature[ids,None]
        return self.history.flatten(1)

    def features(self):
        return torch.cat([self.env.progress_buf[:,None].float(),self.env.state.rigid_body_pos[:,0]],-1)

    def step(self,action):
        self.env.progress_buf+=1
        self.env.state.rigid_body_pos[:,:,0]+=.01
        old=(self.env.state.rigid_body_pos-self.box_center[:,None]).abs()-self.box_half[:,None]
        old=old.clamp_min(0).norm(dim=-1).amin(-1)
        reward=5.-2*torch.exp(-old/.12)
        self.history=torch.cat([self.history[:,1:],self.features()[:,None]],1)
        self.done=torch.tensor([False,True])
        self.info={'original':'untouched','success':self.done.clone()}
        return self.history.flatten(1),reward,self.done,self.info


def run():
    base=ModuleType('warm_task_base');base.WarmRecoveryTask=OriginalTask
    sys.modules['warm_task_base']=base
    pkg=ModuleType('protomotions');assets=ModuleType('protomotions.assets')
    assets.resolve_asset_root=lambda root:root
    sys.modules['protomotions']=pkg;sys.modules['protomotions.assets']=assets
    from warm_task import WarmRecoveryTask
    from foot_clearance import FOOT_NAMES
    with tempfile.TemporaryDirectory(prefix='surface_task_') as directory:
        root=Path(directory);xml=root/'robot.xml'
        xml.write_text('<mujoco><worldbody>'+''.join(
            f'<body name="{name}"><geom type="box" size=".1 .05 .04" pos=".03 0 0"/></body>'
            for name in FOOT_NAMES)+'</worldbody></mujoco>')
        names=['Pelvis',*FOOT_NAMES]
        state=SimpleNamespace(rigid_body_pos=torch.zeros(2,5,3),rigid_body_rot=torch.zeros(2,5,4))
        state.rigid_body_rot[...,3]=1
        env=SimpleNamespace(names=names,progress_buf=torch.zeros(2,dtype=torch.long),state=state,
            robot_config=SimpleNamespace(asset=SimpleNamespace(asset_root=str(root),asset_file_name='robot.xml')),
            simulator=SimpleNamespace(get_robot_state=lambda:state))
        task=WarmRecoveryTask(SimpleNamespace(env=env),{'mode':'evaluate','surface_curriculum':True})
        calls=[]
        task.foot_geometry.register_forward_hook(lambda *args:calls.append(1))
        task.reset();assert len(calls)==1 and task.history.shape==(2,4,20)
        first=task.features().clone();task.features();assert len(calls)==1
        task.strength+=1;task.start_step+=100;task.direction+=1
        assert torch.equal(first,task.features()) and len(calls)==1
        assert torch.equal(first[:,:4],OriginalTask.features(task))
        x,reward,done,info=task.step(torch.randn(2,69))
        assert len(calls)==2 and done is task.done and info is task.info
        assert info['original']=='untouched'
        torch.testing.assert_close(reward,5.-2*torch.exp(-task._surface_gap/.12),atol=1e-6,rtol=0)
        assert torch.all(task._surface_gap<=task._bodypoint_gap)
        assert torch.equal(x.reshape(2,4,20)[:,-1,:4],OriginalTask.features(task))
        before=task.history[1].clone()
        task.reset(torch.tensor([0]));assert len(calls)==3
        assert torch.equal(task.history[1],before)
        assert not task.surface_curriculum.enabled
        assert task.cfg['surface_mjcf_sha256'] and task.cfg['geometry_version']
    print('SURFACE_TASK_CPU_PASSED: one_geometry_sample_per_step; partial_reset; old_feature_prefix; '
          'future_metadata_invariance; exact_same_done_info; only_proximity_replaced; eval_curriculum_disabled')


if __name__=='__main__':run()
