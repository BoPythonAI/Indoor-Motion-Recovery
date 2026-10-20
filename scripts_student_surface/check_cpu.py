"""Actual surface collector -> dataset -> unchanged models -> controller, CPU only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from warm_dependency import install
SURFACE=install()
from contracts import (FEATURE_VERSION,ACTION_TARGET_VERSION,POLICY_KIND,TEACHER_KIND,GEOMETRY_VERSION,
                       file_sha256,validate_checkpoint,validate_runtime)
from controller import WarmStudentController,load_model
from dataset import load_episodes,validate_episode,Episodes,correction_statistics
from model import ChunkPolicy,update_ema
from foot_clearance import FootClearance,FOOT_NAMES
from recovery_runtime import compose_action
from collect_recovery import collect


def reject(fn,word):
    try:fn()
    except ValueError as error:assert word in str(error),str(error)
    else:raise AssertionError('Expected explicit rejection')


class FakeTask:
    feature_version=FEATURE_VERSION
    def __init__(self,xml):
        self.n=2;self.device='cpu';self.names=['Pelvis',*FOOT_NAMES]
        self.geometry=FootClearance(xml,self.names)
        self.rows=[dict(split='train',family='synthetic',motion_id='cpu')]
        self.env=SimpleNamespace(progress_buf=torch.zeros(2,dtype=torch.long),
            motion_manager=SimpleNamespace(motion_ids=torch.zeros(2,dtype=torch.long)))
        self.reset()
    def features(self):
        pos=torch.zeros(2,5,3);pos[:,:,0]=self.state[:,None]*.01
        quat=torch.zeros(2,5,4);quat[...,3]=1
        center=torch.tensor([[.8,0.,0.],[.9,0.,0.]])
        half=torch.tensor([[.1,.2,.2]]).expand(2,3)
        foot=self.geometry.features(self.geometry(pos,quat,center,half),quat[:,0])
        terrain=torch.linspace(-.1,.1,256)[None].expand(2,-1)+self.state[:,None]*.001
        return torch.cat([self.state[:,None],self.env.progress_buf[:,None].float(),terrain,foot],-1)
    def reset(self,ids=None,cases=None):
        self.env.progress_buf.zero_();self.state=torch.zeros(2)
        self.strength=torch.full((2,),.45);self.direction=torch.tensor([[1.,0.],[1.,0.]])
        self.start_step=torch.full((2,),2,dtype=torch.long);self.deadline=torch.full((2,),8,dtype=torch.long)
        self.fallen=torch.zeros(2,dtype=torch.bool);self.violation=self.fallen.clone()
        self.history=self.features()[:,None].repeat(1,4,1);self._pending_prior=None
        return self.history.flatten(1)
    def set_prior(self,prior):self._pending_prior=prior
    def step(self,action):
        prior=self._pending_prior;self._pending_prior=None
        assert prior is not None
        _,total,residual=compose_action(prior.flat_raw,action)
        _,prior_total,_=compose_action(prior.flat_raw,prior.delta)
        active=(self.env.progress_buf>=self.start_step)&(self.env.progress_buf<self.start_step+6)
        self.state+=total.mean(-1)+active.float()*self.strength*.1
        self.env.progress_buf+=1
        self.history=torch.cat([self.history[:,1:],self.features()[:,None]],1)
        done=self.env.progress_buf>=8
        info=dict(base_normalized=prior.flat_raw.tanh(),executed_total=total,executed_residual=residual,
            executed_correction=total-prior_total,prior_total=prior_total,prior_delta=prior.delta,
            actual_correction=total-prior_total,action_target_version=ACTION_TARGET_VERSION,success=done.clone())
        return self.history.flatten(1),torch.zeros(2),done,info


class Prior:
    def __init__(self,task):
        self.task=task;self.calls=0
        self.identity=dict(flat_sha256='a'*64,terrain_sha256='b'*64)
    def current(self):
        self.calls+=1
        flat=torch.zeros(self.task.n,69)
        delta=(.1+self.task.env.progress_buf[:,None].float()*.01).expand(-1,69)
        return SimpleNamespace(flat_raw=flat,terrain_raw=delta.atanh(),delta=delta)


class CleanTeacher:
    def __init__(self,task):self.task=task;self.prior=Prior(task)
    def reset(self,ids=None):self.task._pending_prior=None
    def __call__(self,history):
        prior=self.prior.current();self.task.set_prior(prior)
        correction=(.01+history.reshape(self.task.n,4,-1)[:,-1,0]*.1)[:,None]
        return SimpleNamespace(mean=prior.delta+correction),None


def run():
    torch.set_num_threads(2);torch.manual_seed(73)
    with tempfile.TemporaryDirectory(prefix='student_surface_') as directory:
        root=Path(directory);xml=root/'robot.xml'
        xml.write_text('<mujoco><worldbody>'+''.join(
            f'<body name="{name}"><geom type="box" size=".1 .05 .04" pos=".03 0 0"/></body>'
            for name in FOOT_NAMES)+'</worldbody></mujoco>')
        task=FakeTask(xml);teacher=CleanTeacher(task)
        checkpoint=root/'teacher.pt';checkpoint.write_bytes(b'synthetic surface teacher identity')
        cfg=dict(output=str(root/'episodes'),checkpoint=str(checkpoint),fixed_attempts=2,
            rollin_method='noisyteacher',rollin_noise=.1,feature_version=FEATURE_VERSION,
            takeover_schedule=[dict(name='push_start',after_push_end_steps=-6)],
            teacher_provenance=dict(kind=TEACHER_KIND,geometry_version=GEOMETRY_VERSION,
                surface_mjcf_sha256=file_sha256(xml),surface_mjcf_path=str(xml)),
            stop_reference_version='original_freeze_v5_dynamic',motion_bank_sha256='d'*64,motion_metadata_sha256='e'*64)
        result=collect(task,teacher,cfg)
        assert result['saved']==2
        for path in sorted((root/'episodes').glob('episode_*.pt')):
            episode=torch.load(path,weights_only=False)
            validate_episode(episode)
            assert episode['record_start_control_step']==2 and len(episode['feature'])==6
            assert episode['feature'].shape[-1]==2+256+16
            expected=(.01+episode['feature'].float()[:,0]*.1)[:,None].expand(-1,69)
            torch.testing.assert_close(episode['executed_correction'],expected,atol=8e-5,rtol=0)
            assert episode['executed_correction'].dtype==torch.float32
        dataset,files=load_episodes([str(root/'episodes')],'cpu')
        history,target,mask=dataset.chunk(dataset.valid_indices[[0,5]],6)
        assert mask[0].sum()==6 and mask[1].sum()==1
        assert not torch.equal(history[0].reshape(4,-1)[0],history[0].reshape(4,-1)[-1])
        changed=copy.deepcopy(episode);changed['teacher_provenance']['surface_mjcf_sha256']='f'*64
        reject(lambda:Episodes([(episode,'s'),(changed,'s')]),'Do not mix')
        spec=importlib.util.spec_from_file_location('warm_model_reference',SURFACE.parent/'scripts_student_warm'/'model.py')
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        obs_dim=history.shape[-1];tested=[]
        for kind in ('bc','diffusion'):
            torch.manual_seed(11);model=ChunkPolicy(obs_dim,kind)
            actions=dataset.actions[dataset.valid_indices]
            model.action_mean.copy_(actions.mean(0));model.action_std.copy_(actions.std(0,unbiased=False).clamp_min(.005))
            old=reference.ChunkPolicy(obs_dim,kind);old.load_state_dict(model.state_dict())
            torch.manual_seed(22);loss=model.loss(history,target,mask)
            torch.manual_seed(22);old_loss=old.loss(history,target,mask)
            assert torch.equal(loss,old_loss)
            loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.manual_seed(33);sample=model.sample(history)
            torch.manual_seed(33);assert torch.equal(sample,old.sample(history))
            ema=copy.deepcopy(model);update_ema(ema,model,0)
            state=dict(**dataset.contract,policy_kind=POLICY_KIND,obs_dim=obs_dim,horizon=6,kind=kind,
                model=ema.state_dict(),ema_model=ema.state_dict(),online_model=model.state_dict())
            validate_checkpoint(state)
            legacy=dict(state,policy_kind='terrain_prior_correction_student_v1')
            reject(lambda:validate_checkpoint(legacy),'Legacy student')
            model_path=root/(kind+'.pt');torch.save(state,model_path)
            loaded,saved=load_model(model_path,weights='ema')
            for execute in (1,2):
                replay_task=FakeTask(xml);prior=Prior(replay_task)
                replay_task.history=history.reshape(2,4,-1).clone()
                replay_task.state=replay_task.history[:,-1,0].clone()
                replay_task.env.progress_buf.copy_(replay_task.history[:,-1,1].long())
                deploy=dict(stop_reference_version='original_freeze_v5_dynamic',motion_bank_sha256='1'*64,
                    motion_metadata_sha256='2'*64,geometry_version=GEOMETRY_VERSION,surface_mjcf_sha256=file_sha256(xml))
                validate_runtime(saved,replay_task,prior,deploy)
                reject(lambda:validate_runtime(saved,replay_task,prior,dict(deploy,surface_mjcf_sha256='3'*64)),
                       'Surface geometry changed')
                controller=WarmStudentController(replay_task,loaded,prior,seed=41,execute_steps=execute)
                generator=torch.Generator().manual_seed(41)
                chunk=loaded.sample(history,8,generator)
                first=controller(history)[0].mean
                torch.testing.assert_close(first,controller.last_prior+chunk[:,0],atol=0,rtol=0)
                replay_task.step(first)
                next_history=replay_task.history.flatten(1)
                expected=loaded.sample(next_history,8,generator)[:,0] if execute==1 else chunk[:,1]
                second=controller(next_history)[0].mean
                torch.testing.assert_close(second,controller.last_prior+expected,atol=0,rtol=0)
                assert prior.calls==2
                other=controller.cache[1].clone();controller.reset(torch.tensor([0]))
                assert torch.equal(other,controller.cache[1])
                tested.append(dict(kind=kind,execute_steps=execute,fresh_prior_queries=prior.calls))
        report=dict(status='CPU_PASSED',actual_collector=str(SURFACE/'collect_recovery.py'),
            synthetic_continuations=2,features_per_frame=274,foot_features=16,terrain_features=256,
            labels='float32 executed_total-prior_total from clean actual-state takeover',
            same_model_loss_and_sampling=True,EMA_checked=True,controller_replay=tested,
            legacy_warm_rejected=True,geometry_hash_mixing_rejected=True,simulator_runs=0,optimizer_steps=0)
        Path(__file__).with_name('cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':run()
