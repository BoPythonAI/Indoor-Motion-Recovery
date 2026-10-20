"""CPU checks of schedule-only changes, isolation, partial resets and counters."""
import copy
import json
from types import SimpleNamespace as NS

import torch
from surface_curriculum import SurfaceCurriculum, GROUPS


def task_fixture(n=12000, mode='train', enabled=True):
    ids = torch.arange(n)
    mids = ids % 3
    frames = torch.tensor([70,100,180])
    angle = torch.rand(n)*6.283185307179586
    phase = .25+.4*torch.rand(n)
    strength = torch.where(torch.rand(n)<.5,.45,.75)
    strength[torch.rand(n)<.2]=0
    goal = torch.zeros(n,3); goal[:,0]=ids*20.; goal[:,2]=1
    task = NS(n=n,device='cpu',ids=ids,rows=[dict(split='train')]*3,
        source_frames=frames,goal=goal,box_center=goal+torch.tensor([2.,1.,0.]),
        box_half=torch.tensor([.3,.4,.8]).expand(n,3).clone(),
        start_step=(frames[mids]*phase).long().clamp_min(8),strength=strength,
        direction=torch.stack([angle.cos(),angle.sin(),torch.zeros(n)],-1),
        deadline=frames[mids]+150,history=torch.randn(n,4,3),
        env=NS(motion_manager=NS(motion_ids=mids,motion_times=torch.zeros(n)),
               motion_lib=NS(motion_dt=torch.full((3,),1/30))))
    config=dict(mode=mode,surface_curriculum=enabled)
    return task,config


def snapshot(task):
    return {key:getattr(task,key).clone() for key in
            ('start_step','strength','direction','goal','box_center','box_half','deadline','history')}


def equal_snapshot(a,b,keys=None):
    for key in keys or a:
        assert torch.equal(a[key],b[key]),key


def isolation_checks():
    for mode, enabled in [('evaluate',True),('collect',True),('train',False),('train',{'enabled':False})]:
        task,cfg=task_fixture(32,mode,enabled)
        before=snapshot(task);rng=torch.get_rng_state().clone()
        curriculum=SurfaceCurriculum(task,cfg)
        curriculum.reset_after()
        curriculum.observe_step({},torch.zeros(32,4),torch.zeros(32,dtype=torch.bool))
        equal_snapshot(before,snapshot(task)); assert torch.equal(rng,torch.get_rng_state())
        assert not curriculum.metrics()['enabled']
    task,cfg=task_fixture(32)
    curriculum=SurfaceCurriculum(task,cfg)
    before=snapshot(task);rng=torch.get_rng_state().clone()
    cases=[dict(motion=0,bw=.75,direction=0,phase=.3)]*32
    curriculum.reset_after(cases=cases)
    curriculum.observe_step({'violation':torch.ones(32,dtype=torch.bool)},torch.zeros(32,4),torch.ones(32,dtype=torch.bool))
    equal_snapshot(before,snapshot(task)); assert torch.equal(rng,torch.get_rng_state())
    assert sum(group['steps'] for group in curriculum.metrics()['groups'].values()) == 0
    # Even an already constructed training task cannot reschedule if switched to evaluation.
    task,cfg=task_fixture(32)
    curriculum=SurfaceCurriculum(task,cfg)
    cfg['mode']='evaluate'
    before=snapshot(task);rng=torch.get_rng_state().clone()
    curriculum.reset_after()
    equal_snapshot(before,snapshot(task)); assert torch.equal(rng,torch.get_rng_state())


def distribution_checks():
    torch.manual_seed(444)
    task,cfg=task_fixture()
    before=snapshot(task)
    curriculum=SurfaceCurriculum(task,cfg); curriculum.reset_after()
    after=snapshot(task)
    equal_snapshot(before,after,('strength','goal','box_center','box_half','deadline','history'))
    groups=curriculum.group
    assert torch.equal(groups==0,task.strength==0)
    assert abs(float((groups==0).float().mean())-.2)<.02
    nz=(groups>0).sum()
    for group, probability in [(1,.5),(2,.25),(3,.25)]:
        assert abs(float((groups==group).sum()/nz)-probability)<.02
    untouched=groups<2
    assert torch.equal(task.start_step[untouched],before['start_step'][untouched])
    assert torch.equal(task.direction[groups!=3],before['direction'][groups!=3])
    late=groups>=2; nframes=task.source_frames[task.env.motion_manager.motion_ids[late]]
    assert (task.start_step[late]>=(.85*nframes).long()).all()
    assert (task.start_step[late]<(1.05*nframes+1e-5)).all()
    toward=groups==3
    expected=(task.box_center[toward,:2]-task.box_half[toward,:2])-task.goal[toward,:2]
    expected/=expected.norm(dim=-1,keepdim=True)
    torch.testing.assert_close(task.direction[toward,:2],expected)
    torch.testing.assert_close(task.direction.norm(dim=-1),torch.ones(task.n))
    # Reset a subset only; untouched environments retain schedules and episode IDs.
    previous=snapshot(task);old_groups=groups.clone()
    ids=torch.tensor([2,7,9,21]);mask=torch.ones(task.n,dtype=torch.bool);mask[ids]=False
    curriculum.reset_after(ids)
    for key in previous:
        assert torch.equal(previous[key][mask],snapshot(task)[key][mask]),key
    assert torch.equal(old_groups[mask],curriculum.group[mask])
    return {name:float((groups==i).float().mean()) for i,name in enumerate(GROUPS)}


def fallback_and_deadline_checks():
    torch.manual_seed(23)
    task,cfg=task_fixture(256)
    task.box_center.copy_(task.goal)
    original=task.direction.clone()
    curriculum=SurfaceCurriculum(task,cfg);curriculum.reset_after()
    assert torch.equal(task.direction,original)
    count=int((curriculum.group==3).sum());assert count>0
    assert curriculum.metrics()['groups']['late_toward']['direction_fallbacks']==count
    task,cfg=task_fixture(256)
    task.deadline.fill_(10)
    before=snapshot(task)
    curriculum=SurfaceCurriculum(task,cfg)
    try:
        curriculum.reset_after()
    except ValueError as error:
        assert '66 steps' in str(error)
    else:
        raise AssertionError('Impossible late deadline was silently accepted')
    equal_snapshot(before,snapshot(task))


def metric_checks():
    torch.manual_seed(99)
    task,cfg=task_fixture(128)
    curriculum=SurfaceCurriculum(task,cfg);curriculum.reset_after()
    chosen=torch.tensor([int((curriculum.group==g).nonzero()[0]) for g in range(4)])
    curriculum.active.zero_();curriculum.active[chosen]=True
    gaps=torch.full((task.n,4),.3)
    gaps[chosen]=torch.tensor([.11,.09,.04,.03])[:,None]
    info=dict(violation=torch.zeros(task.n,dtype=torch.bool),external_force=torch.zeros(task.n,3))
    info['external_force'][chosen[2],0]=1
    task.env.motion_manager.motion_times[chosen]=.5
    done=torch.zeros(task.n,dtype=torch.bool);done[chosen[:2]]=True
    before_info=copy.deepcopy(info);before_gaps=gaps.clone();rng=torch.get_rng_state().clone()
    curriculum.observe_step(info,gaps,done)
    for key in info:assert torch.equal(before_info[key],info[key])
    assert torch.equal(gaps,before_gaps) and torch.equal(rng,torch.get_rng_state())
    window1=curriculum.metrics(reset=True)['groups']
    assert window1['nominal']['near10_episode_rate']==0
    assert window1['original_random']['near10_episode_rate']==1
    assert window1['late_random']['completed']==0 and window1['late_random']['near05_step_rate']==1
    gaps.fill_(.3);done[:]=True;info['violation'][chosen[3]]=True
    curriculum.observe_step(info,gaps,done)
    window2=curriculum.metrics()['groups']
    assert window2['late_random']['near05_episode_rate']==1
    assert window2['late_random']['near05_step_rate']==0
    assert window2['late_toward']['forbidden_episode_rate']==1
    assert window1['late_random']['observed_push_starts']==1
    assert window2['late_random']['observed_push_starts']==0
    snapshot_metrics=curriculum.metrics()
    curriculum.observe_step(info,gaps,done)
    assert curriculum.metrics()==snapshot_metrics


def main():
    isolation_checks()
    proportions=distribution_checks()
    fallback_and_deadline_checks()
    metric_checks()
    print(json.dumps(dict(disabled_evaluate_collect_explicit_cases_unchanged=True,
        disabled_and_explicit_cases_rng_unchanged=True,nominal_mask_and_strength_exactly_preserved=True,
        group_proportions=proportions,early_schedule_and_random_directions_preserved=True,
        late_schedule_bounds_and_toward_geometry=True,partial_reset_isolated=True,
        fallback_keeps_nonzero_random_direction=True,impossible_deadline_rejected_before_task_writes=True,
        episode_metrics_survive_window_reset=True,no_duplicate_finished_episodes=True,
        diagnostic_inputs_not_mutated=True,simulator_runs=0),indent=2))


if __name__=='__main__':
    main()
