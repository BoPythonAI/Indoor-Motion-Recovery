"""Batched recovery task and residual PPO. All policy inputs are causal."""
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from protomotions.utils import rotations as rot
from recovery_runtime import base_action, compose_action
from scene_adapter import scene_specs


class ResidualPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim=69, width=256):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(obs_dim,width),nn.ELU(),nn.Linear(width,width),nn.ELU(),nn.Linear(width,action_dim))
        self.critic=nn.Sequential(nn.Linear(obs_dim,width),nn.ELU(),nn.Linear(width,width),nn.ELU(),nn.Linear(width,1))
        nn.init.zeros_(self.actor[-1].weight); nn.init.zeros_(self.actor[-1].bias)
        self.logstd=nn.Parameter(torch.full((action_dim,),math.log(.018)))
        self.obs_dim=obs_dim

    def forward(self, x):
        mu=.3*torch.tanh(self.actor(x))
        return Normal(mu,self.logstd.clamp(math.log(.006),math.log(.08)).exp()), self.critic(x).squeeze(-1)


class RecoveryTask:
    feature_version='v3_terrain_reference'
    def __init__(self,agent,cfg):
        self.agent,self.env,self.cfg=agent,agent.env,cfg
        cfg['feature_version']=self.feature_version
        cfg['reference_placement']='signed_foot_support_per_future_v1'
        cfg['motion_sampling']='equal_source_family'
        self.device=self.env.device
        self.n=self.env.num_envs
        self.ids=torch.arange(self.n,device=self.device)
        self.names=self.env.robot_config.kinematic_info.body_names
        self.feet=[self.names.index(n) for n in ['L_Ankle','L_Toe','R_Ankle','R_Toe']]
        self.nonfeet=[i for i in range(24) if i not in self.feet]
        self.robot=self.env.simulator._robot
        mass=self.robot.root_physx_view.get_masses()
        if not isinstance(mass,torch.Tensor):
            import warp as wp
            mass=wp.to_torch(mass)
        self.mass=mass.sum(-1).to(self.device)
        self.pelvis=self.robot.body_names.index('Pelvis')
        self.rows=json.loads(Path(cfg['motion_metadata']).read_text())
        families=Counter(r['family'] for r in self.rows)
        self.motion_sampling_weights=torch.tensor(
            [1./families[r['family']] for r in self.rows],device=self.device)
        self.motion_sampling_weights/=self.motion_sampling_weights.sum()
        self.source_frames=torch.tensor([r['source_frames'] for r in self.rows],device=self.device)
        self.goal=torch.zeros(self.n,3,device=self.device)
        self.start_step=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.strength=torch.zeros(self.n,device=self.device)
        self.direction=torch.zeros(self.n,3,device=self.device)
        self.deadline=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.fallen=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.stable_count=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.last_total=torch.zeros(self.n,69,device=self.device)
        self.last_residual=torch.zeros_like(self.last_total)
        self.phase_lag=torch.zeros(self.n,device=self.device)
        self.peak_contact=torch.zeros(self.n,device=self.device)
        self.min_height=torch.ones(self.n,device=self.device)*100
        self.prev_distance=torch.zeros(self.n,device=self.device)
        self.history=None
        self.current_bw=cfg.get('max_bw',.45)
        self.last_raw=None
        self.wall_steps=0
        self.obstacle=cfg.get('obstacle',False)
        self.scenes=scene_specs(cfg)
        self.scene_dir=self.scenes[0]['scene_dir'] if self.scenes else None
        self.box_center=torch.zeros(self.n,3,device=self.device)
        self.box_half=torch.tensor([.25,.25,.5],device=self.device).expand(self.n,3).clone()
        self.ground_peak=torch.zeros(self.n,24,device=self.device)
        self.forbidden_peak=torch.zeros(self.n,device=self.device)
        self.violation=torch.zeros(self.n,dtype=torch.bool,device=self.device)
        self.sensing=False
        self.recovery_streak=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.recovery_step=torch.full((self.n,),-1,dtype=torch.long,device=self.device)
        self.target_polygons=[]
        self.episode_forbidden_peak=torch.zeros(self.n,device=self.device)
        self.action_smooth_sum=torch.zeros(self.n,device=self.device)
        self.energy_joules=torch.zeros(self.n,device=self.device)
        if self.scene_dir:
            bounds=[]
            for scene in self.scenes:
                directory=Path(scene['scene_dir']);meta=json.loads((directory/'scene_metadata.json').read_text())
                self.target_polygons.append(torch.tensor(np.load(directory/'scene_arrays.npz')['target_polygon'],device=self.device,dtype=torch.float32))
                bounds.append(next(o for o in meta['objects'] if o['object_id']==scene['forbidden_id'])['bounds'])
            self.scene_bounds=torch.tensor(bounds,device=self.device)
        self.floor_channels=max(1,len(self.scenes))
        original_update=self.env.simulator._scene.update
        def record_contacts(*args,**kwargs):
            original_update(*args,**kwargs)
            if not self.sensing:return
            ground=[]; forbidden=[]
            for name in self.names:
                mat=self.env.simulator._contact_sensor_map[name].data.force_matrix_w
                mat=getattr(mat,'torch',mat)
                ground.append(mat[:,0,:self.floor_channels].norm(dim=-1).sum(-1))
                if mat.shape[2]>self.floor_channels:forbidden.append(mat[:,0,self.floor_channels:].norm(dim=-1).amax(-1))
            self.ground_peak=torch.maximum(self.ground_peak,torch.stack(ground,-1))
            if forbidden:self.forbidden_peak=torch.maximum(self.forbidden_peak,torch.stack(forbidden,-1).amax(-1))
        self.env.simulator._scene.update=record_contacts

    @torch.no_grad()
    def reset(self,ids=None,cases=None):
        ids=self.ids if ids is None else ids
        m=self.env.motion_manager
        if cases is None:
            mids=torch.multinomial(self.motion_sampling_weights,len(ids),replacement=True)
            angle=torch.rand(len(ids),device=self.device)*2*math.pi
            phase=torch.rand(len(ids),device=self.device)*.4+.25
            low=min(self.cfg.get('min_bw',0.),self.current_bw)
            strength=low+torch.rand(len(ids),device=self.device)*(self.current_bw-low)
            if self.cfg.get('force_strengths'):
                levels=torch.tensor(self.cfg['force_strengths'],device=self.device)
                strength=levels[torch.randint(len(levels),(len(ids),),device=self.device)]
            strength[torch.rand(len(ids),device=self.device)<.2]=0
        else:
            mids=torch.tensor([c['motion'] for c in cases],device=self.device)
            angle=torch.tensor([c['direction']*math.pi/2 for c in cases],device=self.device)
            phase=torch.tensor([c['phase'] for c in cases],device=self.device)
            strength=torch.tensor([c['bw'] for c in cases],device=self.device,dtype=torch.float32)
        m.motion_ids[ids]=mids; m.motion_times[ids]=0
        original_spawn=self.env.update_respawn_root_offset_by_env_ids
        if cases is not None and not self.scene_dir:
            def paired_spawn(env_ids,ref_state=None,sample_flat=False):
                offset=torch.zeros(len(env_ids),3,device=self.device)
                offset[:,0]=5+(env_ids%32)*4-ref_state.root_pos[:,0]
                offset[:,1]=5+(env_ids//32)*4-ref_state.root_pos[:,1]
                offset[:,2]=self.env.config.ref_respawn_offset
                self.env.respawn_root_offset[env_ids]=offset
            self.env.update_respawn_root_offset_by_env_ids=paired_spawn
        try:self.obs,_=self.env.reset(ids,disable_motion_resample=True)
        finally:self.env.update_respawn_root_offset_by_env_ids=original_spawn
        lengths=self.env.motion_lib.motion_lengths[mids]
        ref=self.env.motion_lib.get_motion_state(mids,lengths)
        self.goal[ids]=ref.rigid_body_pos[:,0]+self.env.respawn_root_offset[ids]
        if self.scene_dir:
            terminal_offset=self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                ref.rigid_body_pos,env_ids=ids)
            self.goal[ids]=ref.rigid_body_pos[:,0]+terminal_offset[:,0]
        self.deadline[ids]=torch.round(lengths/self.env.dt).long()
        self.start_step[ids]=(self.source_frames[mids]*phase).long().clamp_min(8)
        self.strength[ids]=strength
        # Four world directions; random training directions cover the whole circle.
        self.direction[ids]=torch.stack([angle.cos(),angle.sin(),torch.zeros_like(angle)],-1)
        self.fallen[ids]=False; self.stable_count[ids]=0
        self.last_total[ids]=0; self.last_residual[ids]=0
        self.phase_lag[ids]=0; self.peak_contact[ids]=0; self.min_height[ids]=100
        self.violation[ids]=False
        self.episode_forbidden_peak[ids]=0;self.action_smooth_sum[ids]=0;self.energy_joules[ids]=0
        self.recovery_streak[ids]=0;self.recovery_step[ids]=-1
        s=self.env.simulator.get_robot_state()
        if self.obstacle:
            # Place a fixed physical box to either side of the future path.
            midref=self.env.motion_lib.get_motion_state(mids,self.source_frames[mids]*self.env.dt*.5)
            path=self.goal[ids,:2]-s.rigid_body_pos[ids,0,:2]
            tangent=path/path.norm(dim=-1,keepdim=True).clamp_min(.1)
            side=torch.stack([-tangent[:,1],tangent[:,0]],-1)
            if cases is None:
                sign=torch.where(torch.rand(len(ids),device=self.device)<.5,-1.,1.)
            else: sign=torch.where(torch.arange(len(ids),device=self.device)%2==0,-1.,1.)
            self.box_center[ids]=midref.rigid_body_pos[:,0]+self.env.respawn_root_offset[ids]
            self.box_center[ids,:2]+=side*(sign*.85)[:,None]
            self.box_center[ids,2]=.5+self.env.respawn_root_offset[ids,2]
            pose=torch.zeros(len(ids),7,device=self.device);pose[:,:3]=self.box_center[ids];pose[:,6]=1
            self.env.simulator._scene['forbidden'].write_root_pose_to_sim(pose,env_ids=ids)
        if self.scene_dir:
            scene_id=self.env.v4_row_scene[mids];bounds=self.scene_bounds[scene_id]
            self.box_center[ids]=bounds.mean(1)+self.env.simulator._scene.env_origins[ids]+self.env.v4_room_offsets[scene_id]
            self.box_half[ids]=(bounds[:,1]-bounds[:,0])/2
        self.prev_distance[ids]=(s.rigid_body_pos[ids,0,:2]-self.goal[ids,:2]).norm(dim=-1)
        feature=self.features()
        if self.history is None: self.history=feature[:,None,:].repeat(1,4,1)
        else: self.history[ids]=feature[ids,None,:]
        return self.history.flatten(1)

    @torch.no_grad()
    def features(self):
        s=self.env.simulator.get_robot_state()
        qinv=rot.calc_heading_quat_inv(s.rigid_body_rot[:,0],True)
        def local(v): return rot.quat_rotate(qinv,v,True)
        obs=self.env.get_obs()
        trunk=self.agent.model._actor.mu
        normal=trunk.norm(torch.cat([obs[k] for k in trunk.config.in_keys],-1))
        forces=s.rigid_body_contact_forces/(self.mass[:,None,None]*9.81)
        force_heading=qinv[:,None].expand(-1,24,-1).reshape(-1,4)
        forces=rot.quat_rotate(force_heading,forces.reshape(-1,3),True).reshape(self.n,24,3)
        up=rot.quat_rotate_inverse(s.rigid_body_rot[:,0],torch.tensor([0.,0.,1.],device=self.device).expand(self.n,3),True)
        err=self.env.context.mimic.ref_state.rigid_body_pos[:,0]-s.rigid_body_pos[:,0]
        remaining=(self.deadline-self.env.progress_buf).float()*self.env.dt
        # Geometry slots are disabled in the first, flat-ground curriculum stage.
        geometry=torch.zeros(self.n,97,device=self.device)
        if self.obstacle or self.scene_dir:
            closest=torch.maximum(torch.minimum(s.rigid_body_pos,self.box_center[:,None]+self.box_half[:,None]),self.box_center[:,None]-self.box_half[:,None])
            delta=closest-s.rigid_body_pos
            dist=delta.norm(dim=-1)
            q=qinv[:,None].expand(-1,24,-1).reshape(-1,4)
            direction=rot.quat_rotate(q,delta.reshape(-1,3),True).reshape(self.n,72)
            geometry=torch.cat([dist.clamp(max=3)/3,direction.clamp(-3,3)/3,torch.ones(self.n,1,device=self.device)],-1)
        return torch.cat([normal,s.dof_pos/math.pi,(s.dof_vel/10).clamp(-5,5),
                          forces.flatten(1).clamp(-3,3),up,local(err),
                          local(self.goal-s.rigid_body_pos[:,0])/3,
                          torch.stack([remaining/10,self.phase_lag/2],-1),
                          self.last_total,geometry],-1).clamp(-10,10)

    @torch.no_grad()
    def step(self,residual):
        env=self.env
        if self.cfg.get('measure_latency'):
            torch.cuda.synchronize();base_start=time.perf_counter()
        base_raw=base_action(self.agent,env.get_obs())
        raw,total,executed=compose_action(base_raw,residual)
        if self.cfg.get('measure_latency'):
            torch.cuda.synchronize();base_ms=(time.perf_counter()-base_start)*1000
        self.last_raw=raw
        active=(env.progress_buf>=self.start_step)&(env.progress_buf<self.start_step+6)
        f=self.direction*(self.strength*self.mass*9.81*active)[:,None]
        self.robot.permanent_wrench_composer.reset()
        self.robot.set_external_force_and_torque(f[:,None,:],torch.zeros_like(f[:,None,:]),body_ids=[self.pelvis],is_global=True)
        self.ground_peak.zero_();self.forbidden_peak.zero_();self.sensing=True
        try:obs,_,_,_,_=env.step(raw)
        finally:self.sensing=False
        s=env.simulator.get_robot_state()
        ref=env.context.mimic.ref_state
        height=s.rigid_body_pos[:,0,2]-env.respawn_root_offset[:,2]
        if self.scene_dir:
            height=s.rigid_body_pos[:,0,2]-env.terrain.get_ground_heights(s.rigid_body_pos[:,0]).flatten()
        force=s.rigid_body_contact_forces.norm(dim=-1)
        nonfoot_force=self.ground_peak[:,self.nonfeet].amax(-1)
        up=1-2*(s.rigid_body_rot[:,0,:2]**2).sum(-1)
        # Ignore harmless limb brush forces; >20 N non-foot ground support is a fall.
        fallen=(height<.50)|(up<.35)|((nonfoot_force>20)&(env.progress_buf>4))
        self.fallen |= fallen
        violation=self.forbidden_peak>2.
        self.violation |= violation
        self.episode_forbidden_peak=torch.maximum(self.episode_forbidden_peak,self.forbidden_peak)
        self.peak_contact=torch.maximum(self.peak_contact,nonfoot_force)
        self.min_height=torch.minimum(self.min_height,height)
        distance=(s.rigid_body_pos[:,0,:2]-self.goal[:,:2]).norm(dim=-1)
        local_error=((s.rigid_body_pos-s.rigid_body_pos[:,:1])-(ref.rigid_body_pos-ref.rigid_body_pos[:,:1])).square().mean((1,2))
        root_error=(s.rigid_body_pos[:,0,:2]-ref.rigid_body_pos[:,0,:2]).square().sum(-1)
        progress=(self.prev_distance-distance).clamp(-.2,.2)
        smooth=(executed-self.last_residual).square().mean(-1)
        self.action_smooth_sum+=(total-self.last_total).square().mean(-1)
        torque=env.simulator.get_dof_forces().dof_forces
        power=(torque*s.dof_vel).abs().sum(-1)
        self.energy_joules+=power*env.dt
        reward= .5 + torch.exp(-20*local_error) + .5*up.clamp(0,1) + 8*progress + .5*torch.exp(-2*root_error)
        reward-= .5*executed.square().mean(-1)+2*smooth
        reward-=self.cfg.get('power_weight',0.)*power/(self.mass*9.81)
        reward-=20*fallen.float()
        reward-=20*violation.float()
        if self.obstacle or self.scene_dir:
            outside=(s.rigid_body_pos-self.box_center[:,None]).abs()-self.box_half[:,None]
            separation=outside.clamp_min(0).norm(dim=-1).amin(-1)
            reward-=2*torch.exp(-separation/.12)
        stable=(distance<.35)&(height>.65)&(up>.85)
        if not self.scene_dir:
            stable &= s.rigid_body_ang_vel[:,0].norm(dim=-1)<1.5
        target_distance=distance
        if self.scene_dir:
            scene_id=env.v4_row_scene[env.motion_manager.motion_ids]
            p=s.rigid_body_pos[:,0,:2]-env.simulator._scene.env_origins[:,:2]-env.v4_room_offsets[scene_id,:2]
            for k,a in enumerate(self.target_polygons):
                b=a.roll(-1,0);v=b-a;rel=p[:,None]-a[None]
                t=(rel*v[None]).sum(-1)/v.square().sum(-1).clamp_min(1e-10)
                closest=a[None]+t.clamp(0,1)[:,:,None]*v[None]
                dist=(p[:,None]-closest).norm(dim=-1).amin(-1)
                cross=v[None,:,0]*rel[:,:,1]-v[None,:,1]*rel[:,:,0]
                inside=(cross>=0).all(-1)|(cross<=0).all(-1)
                target_distance=torch.where(scene_id==k,torch.where(inside,0.,dist),target_distance)
            chest=s.rigid_body_pos[:,self.names.index('Chest')]-s.rigid_body_pos[:,0]
            torso_up=chest[:,2]/chest.norm(dim=-1).clamp_min(1e-6)
            stable &= (target_distance<=.8)&(s.rigid_body_vel[:,0,:2].norm(dim=-1)<=.5)&(s.rigid_body_vel[:,0,2].abs()<=.5)&(torso_up>=math.cos(math.pi/4))
            reward-=self.cfg.get('stationary_weight',0.)*(distance<.5).float()*s.rigid_body_vel[:,0].square().sum(-1)
        torso=s.rigid_body_pos[:,self.names.index('Chest')]-s.rigid_body_pos[:,0]
        torso_up=torso[:,2]/torso.norm(dim=-1).clamp_min(1e-6)
        balanced=(height>.65)&(up>.90)&(s.rigid_body_vel[:,0,2].abs()<=.5)&(torso_up>=math.cos(math.pi/4))&~fallen&~violation
        post_push=env.progress_buf>self.start_step+6
        self.recovery_streak=torch.where(balanced&post_push,self.recovery_streak+1,torch.zeros_like(self.recovery_streak))
        first=(self.recovery_streak>=15)&(self.recovery_step<0)
        self.recovery_step=torch.where(first,env.progress_buf,self.recovery_step)
        eligible=(env.progress_buf>self.start_step+36)&(env.progress_buf>=self.source_frames[env.motion_manager.motion_ids])
        self.stable_count=torch.where(stable&eligible,self.stable_count+1,torch.zeros_like(self.stable_count))
        success=(self.stable_count>=(30 if self.scene_dir else 15))&~self.fallen&~self.violation
        timeout=env.progress_buf>=self.deadline
        done=fallen|violation|success|timeout
        # Successful termination can replace the discounted value of continued
        # nominal tracking (2.5 / (1 - 0.99) = 250), avoiding an incentive to stall.
        reward+=self.cfg.get('success_bonus',10.)*success.float()
        self.prev_distance=distance
        self.last_total=total
        self.last_residual=executed
        # Causal, bounded phase delay shared by the four compared methods.
        if self.cfg.get('adaptive_phase',False):
            lagging=(root_error>.09)|(up<.85)
            delay=torch.minimum(lagging.float()*env.dt*.5,(2-self.phase_lag).clamp_min(0))
            self.phase_lag+=delay
            env.motion_manager.motion_times-=delay
            env._current_context=None
            env.compute_observations(context=env.context)
        feature=self.features()
        self.history=torch.cat([self.history[:,1:],feature[:,None]],dim=1)
        self.wall_steps+=1
        info={'success':success,'fallen':fallen,'timeout':timeout,'distance':distance,'height':height,
              'base_normalized':base_raw.tanh(),'executed_residual':executed,'executed_total':total,
              'nonfoot_force':nonfoot_force,'strength':self.strength.clone()}
        info['forbidden_force']=self.forbidden_peak.clone()
        info['violation']=violation
        info['external_force']=f
        info['target_polygon_distance']=target_distance
        info['root_up']=up
        info['root_speed_xy']=s.rigid_body_vel[:,0,:2].norm(dim=-1)
        info['root_vz']=s.rigid_body_vel[:,0,2]
        info['root_angular_speed']=s.rigid_body_ang_vel[:,0].norm(dim=-1)
        if self.cfg.get('measure_latency'):info['base_inference_ms']=base_ms
        return self.history.flatten(1),reward,done,info


def save_checkpoint(path,policy,opt,update,cfg,seconds):
    tmp=path.with_suffix('.tmp')
    torch.save({'policy':policy.state_dict(),'optimizer':opt.state_dict() if opt else None,
                'obs_dim':policy.obs_dim,'update':update,'config':cfg,'training_seconds':seconds},tmp)
    tmp.replace(path)


def evaluate(task,policy,cfg):
    cases=[]
    for mid in range(len(task.rows)):
        for bw in cfg.get('strengths',[0,.45,.75]):
            for direction in range(4):
                for phase in [.30,.60]:
                    cases.append(dict(motion=mid,bw=bw,direction=direction,phase=phase))
    if cfg.get('cases'):cases=cfg['cases']
    rows=[];latencies=[];full_latencies=[]
    traces=[]
    for start in range(0,len(cases),task.n):
        if hasattr(policy,'reset'):policy.reset()
        batch=cases[start:start+task.n]; real=len(batch)
        padded=batch+[batch[-1]]*(task.n-real)
        x=task.reset(cases=padded)
        live=torch.ones(task.n,dtype=torch.bool,device=task.device)
        for step in range(int(task.deadline.max())+1):
            with torch.no_grad():
                if cfg.get('measure_latency'):
                    torch.cuda.synchronize();latency_start=time.perf_counter()
                residual=torch.zeros(task.n,69,device=task.device) if policy is None else policy(x)[0].mean
                if cfg.get('measure_latency'):
                    torch.cuda.synchronize();latencies.append((time.perf_counter()-latency_start)*1000)
                x,_,done,info=task.step(residual)
                if cfg.get('measure_latency'):full_latencies.append(latencies[-1]+info['base_inference_ms'])
                record_id=cfg.get('record_env',0)
                if cfg.get('record') and bool(live[record_id]):
                    state=task.env.simulator.get_robot_state()
                    traces.append({'step':step,'body_xyz':state.rigid_body_pos[record_id].cpu().tolist(),
                                   'body_quat_xyzw_w':state.rigid_body_rot[record_id].cpu().tolist(),
                                   'body_names':task.names,'root_xyz':state.rigid_body_pos[record_id,0].cpu().tolist(),
                                   'root_velocity':state.rigid_body_vel[record_id,0].cpu().tolist(),
                                   'root_angular_velocity':state.rigid_body_ang_vel[record_id,0].cpu().tolist(),
                                   'target_polygon_distance':float(info['target_polygon_distance'][record_id]),
                                   'stable_count':int(task.stable_count[record_id]),
                                   'goal_distance':float(info['distance'][record_id]),'forbidden_force_N':float(info['forbidden_force'][record_id]),
                                   'external_force_xyz_N':info['external_force'][record_id].cpu().tolist(),
                                   'ground_forces_N':task.ground_peak[record_id].cpu().tolist(),
                                   'action':info['executed_total'][record_id].cpu().tolist()})
            finish=(done&live).nonzero(as_tuple=True)[0]
            for j in finish.tolist():
                if j<real:
                    rows.append(dict(**batch[j],motion_id=task.rows[batch[j]['motion']]['motion_id'],
                                     success=bool(info['success'][j]),fallen=bool(task.fallen[j]),
                                     forbidden_violation=bool(task.violation[j]),
                                     recovery_time_s=float((task.recovery_step[j]-task.start_step[j]-6)*task.env.dt) if task.recovery_step[j]>=0 and batch[j]['bw']>0 else None,
                                     steps=step+1,goal_distance=float(info['distance'][j]),
                                     target_polygon_distance=float(info['target_polygon_distance'][j]),
                                     final_root_up=float(info['root_up'][j]),
                                     final_root_speed_xy=float(info['root_speed_xy'][j]),
                                     final_root_vz=float(info['root_vz'][j]),
                                     final_root_angular_speed=float(info['root_angular_speed'][j]),
                                     stable_count=int(task.stable_count[j]),
                                     stable_goal_reached=bool(task.stable_count[j]>0),
                                     peak_forbidden_force_N=float(task.episode_forbidden_peak[j]),
                                     mean_action_delta_squared=float(task.action_smooth_sum[j]/(step+1)),
                                     mechanical_energy_J=float(task.energy_joules[j]),
                                     min_height=float(task.min_height[j]),peak_nonfoot_force=float(task.peak_contact[j])))
            live &= ~done
            # Finish all simulated slots without recycling them into the same case.
            if not live.any(): break
        print('V4_EVAL_BATCH',len(rows),'/',len(cases),flush=True)
    summary={str(bw):{'success':sum(r['success'] for r in rows if r['bw']==bw),
                         'total':sum(r['bw']==bw for r in rows),
                         'falls':sum(r['fallen'] for r in rows if r['bw']==bw)} for bw in cfg.get('strengths',[0,.45,.75])}
    p=Path(cfg['output']);p.mkdir(exist_ok=True,parents=True)
    stage='scannet_recovery' if task.scene_dir else 'box_recovery' if task.obstacle else 'flat_recovery'
    report={'stage':stage,'summary':summary,'episodes':rows}
    if latencies:report['residual_model_latency_ms']={'p50':float(np.percentile(latencies,50)),'p95':float(np.percentile(latencies,95)),'num_envs':task.n}
    if full_latencies:report['tracker_plus_residual_model_latency_ms']={'p50':float(np.percentile(full_latencies,50)),'p95':float(np.percentile(full_latencies,95)),'num_envs':task.n}
    (p/'evaluation.json').write_text(json.dumps(report,indent=2))
    if traces:(p/'trace.json').write_text(json.dumps({'config':cfg,'scene_origin':task.env.simulator._scene.env_origins[cfg.get('record_env',0)].cpu().tolist(),'records':traces}))
    print('V4_COMPLETE',json.dumps(summary),flush=True)
    return summary


def validate_in_place(task,policy,cfg,update):
    """Use the same simulator for a separate validation motion bank, then reset.

    No rollout spans the switch, and the original training bank is restored in
    a finally block. This avoids another large simulator on the same GPU.
    """
    lib=task.env.motion_lib
    bank=torch.load(cfg['validation_motion'],map_location=task.device,weights_only=False)
    old={k:getattr(lib,k) for k in bank if hasattr(lib,k)}
    original_update=task.env.simulator._scene.update
    valcfg=dict(cfg,mode='evaluate',motion_metadata=cfg['validation_metadata'],
                output=str(Path(cfg['output'])/f'validation_{update:04d}'),strengths=[0,.45,.75])
    try:
        for k,v in bank.items():
            if hasattr(lib,k):setattr(lib,k,v)
        valtask=RecoveryTask(task.agent,valcfg)
        return evaluate(valtask,policy,valcfg)
    finally:
        task.env.simulator._scene.update=original_update
        for k,v in old.items():setattr(lib,k,v)
        task.reset()


def train_or_evaluate(agent,cfg):
    task=RecoveryTask(agent,cfg)
    x=task.reset()
    policy=ResidualPolicy(x.shape[-1]).to(task.device)
    if cfg.get('critic_lr'):
        opt=torch.optim.Adam([{'params':list(policy.actor.parameters())+[policy.logstd],'lr':cfg.get('lr',1e-4)},
                              {'params':policy.critic.parameters(),'lr':cfg['critic_lr']}])
    else:opt=torch.optim.Adam(policy.parameters(),lr=cfg.get('lr',1e-4))
    initial_update=0
    if cfg.get('checkpoint'):
        state=torch.load(cfg['checkpoint'],map_location=task.device,weights_only=False)
        if cfg['mode'] in ['evaluate','collect']:
            assert state['config'].get('feature_version','v1_world_contact')==task.feature_version,'Teacher feature convention mismatch'
        policy.load_state_dict(state['policy'])
        initial_update=state.get('update',0) if cfg.get('resume') else 0
        if cfg.get('resume') and state.get('optimizer'): opt.load_state_dict(state['optimizer'])
    if cfg['mode']=='evaluate':
        if cfg.get('method') in ['bc','diffusion']:
            from student import StudentController
            policy=StudentController(cfg['student_checkpoint'],task.device,cfg.get('seed',17),cfg.get('execute_steps',1))
            assert policy.feature_version==task.feature_version,'Student feature convention mismatch'
            assert policy.model.kind==cfg['method'],'Checkpoint controller type does not match the requested method'
        return evaluate(task,None if cfg.get('method')=='base' else policy,cfg)
    if cfg['mode']=='collect':
        if not cfg.get('checkpoint'):raise ValueError('A validated teacher checkpoint is required')
        from collect import collect
        return collect(task,policy,cfg)
    out=Path(cfg['output']);out.mkdir(exist_ok=True,parents=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    stats=[]; started=time.perf_counter(); horizon=cfg.get('horizon',32)
    recent=[]
    for update in range(initial_update,cfg.get('updates',1000)):
        task.current_bw=min(cfg.get('max_bw',.75),cfg.get('start_bw',.15)+(update-initial_update)*cfg.get('curriculum_rate',.001))
        xs=[];acts=[];logs=[];values=[];rewards=[];dones=[]
        with torch.no_grad():
            for t in range(horizon):
                dist,v=policy(x); action=dist.sample(); log=dist.log_prob(action).sum(-1)
                xs.append(x);acts.append(action);logs.append(log);values.append(v)
                x,r,done,info=task.step(action)
                rewards.append(r);dones.append(done)
                if done.any():
                    ids=done.nonzero(as_tuple=True)[0]
                    recent.extend(info['success'][ids].float().cpu().tolist())
                    recent=recent[-512:]
                    x=task.reset(ids)
            next_v=policy(x)[1]
            vs=torch.stack(values); rs=torch.stack(rewards); ds=torch.stack(dones)
            adv=torch.zeros_like(rs); carry=torch.zeros(task.n,device=task.device)
            for t in reversed(range(horizon)):
                nv=next_v if t==horizon-1 else vs[t+1]
                keep=(~ds[t]).float()
                carry=rs[t]+.99*nv*keep-vs[t]+.99*.95*keep*carry
                adv[t]=carry
            returns=(adv+vs).flatten(); adv=adv.flatten(); adv=(adv-adv.mean())/adv.std().clamp_min(1e-6)
            bx=torch.stack(xs).flatten(0,1);ba=torch.stack(acts).flatten(0,1);bl=torch.stack(logs).flatten()
        losses=[];kls=[];stop_policy_update=False
        for epoch in range(cfg.get('ppo_epochs',4)):
            perm=torch.randperm(len(bx),device=task.device)
            for ids in perm.split(cfg.get('batch_size',4096)):
                dist,v=policy(bx[ids]);log=dist.log_prob(ba[ids]).sum(-1)
                ratio=(log-bl[ids]).exp()
                kl=((ratio-1)-(log-bl[ids])).mean().detach()
                # Stop before another minibatch can amplify a policy jump.
                if kl>cfg.get('target_kl',.03):
                    stop_policy_update=True
                    break
                pg=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                value=.5*(v-returns[ids]).square().mean()
                loss=pg+value*.5
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),1.);opt.step()
                losses.append(float(loss));kls.append(float(kl))
            if stop_policy_update:break
        for _ in range(cfg.get('critic_epochs',0)):
            for ids in torch.randperm(len(bx),device=task.device).split(cfg.get('batch_size',4096)):
                value_loss=.5*(policy.critic(bx[ids]).squeeze(-1)-returns[ids]).square().mean()
                opt.zero_grad(set_to_none=True);value_loss.backward()
                nn.utils.clip_grad_norm_(policy.critic.parameters(),1.);opt.step()
        elapsed=time.perf_counter()-started
        record=dict(update=update+1,seconds=elapsed,bw=task.current_bw,reward=float(rs.mean()),
                    recent_success=float(np.mean(recent)) if recent else None,episodes=len(recent),
                    loss=float(np.mean(losses)),kl=float(np.mean(kls)),
                    residual_abs=float(ba.abs().mean()),std=float(policy.logstd.exp().mean()),
                    value_mean=float(vs.mean()),return_mean=float(returns.mean()),
                    explained_variance=float(1-(returns-vs.flatten()).var()/returns.var().clamp_min(1e-6)),
                    transitions_per_second=(update-initial_update+1)*horizon*task.n/elapsed)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if update%10==0:print('V4_TRAIN',json.dumps(record),flush=True)
        if (update+1)%cfg.get('save_every',50)==0:
            save_checkpoint(out/'last.pt',policy,opt,update+1,cfg,elapsed)
        if cfg.get('snapshot_every') and (update+1)%cfg['snapshot_every']==0:
            save_checkpoint(out/f'update_{update+1:04d}.pt',policy,opt,update+1,cfg,elapsed)
        if cfg.get('validation_motion') and (update+1)%cfg.get('validate_every',100)==0:
            summary=validate_in_place(task,policy,cfg,update+1)
            save_checkpoint(out/f'update_{update+1:04d}.pt',policy,opt,update+1,cfg,time.perf_counter()-started)
            x=task.history.flatten(1)
            rates=[summary[str(bw)]['success']/summary[str(bw)]['total'] for bw in [0,.45,.75]]
            if rates[0]>=.95 and rates[1]>=.95 and rates[2]>=.85:
                print('V4_TEACHER_VALIDATED',json.dumps(rates),flush=True)
                break
        if elapsed>cfg.get('max_hours',8)*3600:break
    save_checkpoint(out/'last.pt',policy,opt,update+1,cfg,time.perf_counter()-started)
    print('V4_TRAIN_COMPLETE',str(out/'last.pt'),flush=True)
