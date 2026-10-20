"""CPU-only checks for original goal, timing, partial reset and shared features."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import torch

from motion_reference import file_sha256
from reference_goal_contract import PLAN_VERSION, EXPERIMENTAL_STATUS, apply_original_goal, validate_plan_status
from prepare_planned_evaluation import prepare
from surface_contract import POLICY_KIND, FEATURE_VERSION


def fixture(root, single=False):
    count = 1 if single else 2
    rows = [dict(motion_id=f'source{i}',source_frames=70+i*10,scene_dir=str(root/'scene')) for i in range(count)]
    metadata=root/'metadata.json';metadata.write_text(json.dumps(rows))
    original=root/'original.pt'
    planned=root/'planned.pt'
    counts=torch.tensor([220+i*10 for i in range(count)])
    starts=torch.cat([torch.zeros(1,dtype=torch.long),counts.cumsum(0)[:-1]])
    raw=torch.tensor([[1.1234567+i,2.234567+i,.8+i*.01] for i in range(count)])[:,None,:].repeat(1,24,1)
    raw_plan=raw.clone();raw_plan[:,:,1]-=.08
    scene_translation=torch.tensor([[2.345678+i,-.1234567,0.] for i in range(count)])
    scene_offset=scene_translation.clone();scene_offset[:,2]=.125+torch.arange(count)*.01
    bank=dict(gts=torch.cat([raw[i:i+1].repeat(int(counts[i]),1,1) for i in range(count)]),
        motion_num_frames=counts,length_starts=starts,motion_dt=torch.full((count,),1/30),
        motion_lengths=(counts-1).float()/30)
    planned_bank=dict(bank,gts=torch.cat([raw_plan[i:i+1].repeat(int(counts[i]),1,1) for i in range(count)]))
    torch.save(bank,original);torch.save(planned_bank,planned)
    sources=[]
    for i,row in enumerate(rows):
        sources.append(dict(source_id=row['motion_id'],status='PLANNED_PENDING_PHYSICS',
            source_frames=row['source_frames'],total_frames=int(counts[i]),motion_dt=float(bank['motion_dt'][i]),
            motion_length=float(bank['motion_lengths'][i]),
            original_goal_world_in_scene=(raw[i,0]+scene_offset[i]).tolist(),
            planned_reference_goal_world_in_scene=(raw_plan[i,0]+scene_offset[i]).tolist()))
    manifest=dict(plan_version=PLAN_VERSION,status='PLANNED_PENDING_PHYSICS',original_bank=str(original),
        original_bank_sha256=file_sha256(original),planned_bank=str(planned),planned_bank_sha256=file_sha256(planned),
        original_metadata=str(metadata),original_metadata_sha256=file_sha256(metadata),sources=sources,mjcf_sha256='a'*64)
    path=root/'plan.json';path.write_text(json.dumps(manifest))
    cfg=dict(mode='evaluate',output=str(root/'out'),motion_bank=str(planned),motion_metadata=str(metadata),
        stop_reference_version='original_freeze_v5_dynamic',reference_planning_version=PLAN_VERSION,
        reference_plan_manifest=str(path),surface_mjcf_sha256='a'*64)
    n=4; mids=torch.arange(n)%count
    lib=NS(**planned_bank)
    def frame_blend(motion_ids,times):
        frame=lib.motion_num_frames[motion_ids]-1
        blend=((times-frame*lib.motion_dt[motion_ids])/lib.motion_dt[motion_ids]).clamp(0.,1.)
        return frame,frame,blend
    lib._calc_frame_blend_from_id_and_time=frame_blend
    env=NS(dt=1/30,motion_lib=lib,motion_manager=NS(motion_ids=mids),
        v4_row_scene=torch.arange(count),v4_room_offsets=torch.tensor([[30.*i,0.,0.] for i in range(count)]),
        simulator=NS(_scene=NS(env_origins=torch.tensor([[20.*i,0.,0.] for i in range(n)]))))
    env.respawn_root_offset=env.simulator._scene.env_origins+env.v4_room_offsets[mids]+scene_translation[mids]
    def terrain_offset(positions,env_ids):
        offset=torch.zeros_like(positions)
        offset[:,:,:2]=env.respawn_root_offset[env_ids,None,:2]
        offset[:,:,2]=(.125+env.motion_manager.motion_ids[env_ids]*.01)[:,None]
        return offset
    env.get_spawn_to_ref_pose_offset_with_terrain_height_correction=terrain_offset
    task=NS(cfg=cfg,env=env,rows=rows,device='cpu',n=n,ids=torch.arange(n),scene_dir=str(root/'scene'),
        source_frames=torch.tensor([s['source_frames'] for s in sources]),goal=torch.zeros(n,3),
        deadline=torch.zeros(n,dtype=torch.long),prev_distance=torch.zeros(n),
        history=torch.full((n,4,3),-100.),actual_root=torch.zeros(n,3),
        fixture_raw=raw,fixture_scene_translation=scene_translation)
    return task,manifest,path


def fake_original_reset(task,manifest,ids):
    mids=task.env.motion_manager.motion_ids[ids]
    env=task.env;lib=env.motion_lib
    env.respawn_root_offset[ids]=env.simulator._scene.env_origins[ids]+env.v4_room_offsets[mids]+task.fixture_scene_translation[mids]
    raw=lib.gts[lib.length_starts[mids]+lib.motion_num_frames[mids]-1]
    _,_,blend=lib._calc_frame_blend_from_id_and_time(mids,lib.motion_lengths[mids])
    blend=blend[:,None,None];raw=(1.-blend)*raw+blend*raw
    task.goal[ids]=raw[:,0]+env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(raw,env_ids=ids)[:,0]
    task.deadline[ids]=torch.round(task.env.motion_lib.motion_lengths[mids]/task.env.dt).long()
    deadline=task.deadline.clone();frames=task.source_frames.clone();rng=torch.get_rng_state().clone()
    apply_original_goal(task,ids,mids)
    assert torch.equal(deadline,task.deadline) and torch.equal(frames,task.source_frames)
    assert torch.equal(rng,torch.get_rng_state())
    # Mirrors the ordering after the one hook in the original task. Every mode
    # uses this same reset, including student imports and collector partial resets.
    task.prev_distance[ids]=(task.goal[ids,:2]-task.actual_root[ids,:2]).norm(dim=-1)
    feature=task.goal-task.actual_root
    task.history[ids]=feature[ids,None,:]


def check_one(root,mode):
    task,manifest,path=fixture(root);task.cfg['mode']=mode
    fake_original_reset(task,manifest,task.ids)
    mids=task.env.motion_manager.motion_ids
    translation=task.env.simulator._scene.env_origins+task.env.v4_room_offsets[mids]
    raw=task.fixture_raw[mids]
    _,_,blend=task.env.motion_lib._calc_frame_blend_from_id_and_time(mids,task.env.motion_lib.motion_lengths[mids])
    blend=blend[:,None,None];raw=(1.-blend)*raw+blend*raw
    expected=raw[:,0]+task.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(raw,env_ids=task.ids)[:,0]
    assert torch.equal(task.goal,expected)
    assert torch.equal(task.history,expected[:,None,:].expand(-1,4,-1))
    assert not torch.equal(task.planned_reference_goal,task.original_task_goal)
    previous=task.goal.clone();history=task.history.clone();old_plan=task.planned_reference_goal.clone()
    ids=torch.tensor([1,3]);task.env.motion_manager.motion_ids[ids]=0
    fake_original_reset(task,manifest,ids)
    keep=torch.tensor([0,2])
    assert torch.equal(task.goal[keep],previous[keep]) and torch.equal(task.history[keep],history[keep])
    assert torch.equal(task.planned_reference_goal[keep],old_plan[keep])
    logs=[json.loads(line) for line in (root/'out/reference_goal_resets.jsonl').read_text().splitlines()]
    assert len(logs)==2 and logs[1]['env_ids']==[1,3]
    # Wrong placement/deadline and altered source timing must fail, not move goal.
    task.deadline[ids]+=1
    before=task.goal.clone()
    try:apply_original_goal(task,ids,task.env.motion_manager.motion_ids[ids])
    except ValueError as error:assert 'deadline' in str(error)
    else:raise AssertionError('Changed deadline accepted')
    assert torch.equal(task.goal,before)


def main():
    with tempfile.TemporaryDirectory(prefix='planned_goal_cpu_') as directory:
        root=Path(directory).resolve()
        for mode in ('evaluate','collect','train'):
            folder=root/mode;folder.mkdir();check_one(folder,mode)
        folder=root/'identity';folder.mkdir();task,manifest,path=fixture(folder)
        bank=torch.load(manifest['original_bank'],weights_only=False)
        torch.save(bank,manifest['planned_bank']);task.env.motion_lib.gts=bank['gts']
        manifest['planned_bank_sha256']=file_sha256(manifest['planned_bank'])
        manifest['status']='IDENTITY_ORIGINAL_TERMINAL_GEOMETRY_CLEAR'
        for source in manifest['sources']:
            source['status']=manifest['status']
            source['planned_reference_goal_world_in_scene']=source['original_goal_world_in_scene']
        path.write_text(json.dumps(manifest))
        fake_original_reset(task,manifest,task.ids)
        assert torch.equal(task.goal,task.planned_reference_goal)
        assert torch.equal(task.history,task.planned_reference_goal[:,None,:].expand(-1,4,-1))
        mids=task.env.motion_manager.motion_ids
        regrouped=torch.tensor([s['original_goal_world_in_scene'] for s in manifest['sources']])[mids]+(
            task.env.simulator._scene.env_origins+task.env.v4_room_offsets[mids])
        assert not torch.equal(regrouped,task.goal),'Fixture must expose float32 non-associativity'
        folder=root/'rejected';folder.mkdir();task,manifest,path=fixture(folder)
        manifest['status']='REJECTED_NO_GEOMETRIC_SOLUTION';path.write_text(json.dumps(manifest))
        try:fake_original_reset(task,manifest,task.ids)
        except ValueError as error:assert 'not deployable' in str(error)
        else:raise AssertionError('Rejected geometry accepted')
        folder=root/'experimental';folder.mkdir();task,manifest,path=fixture(folder)
        for entry in [manifest]+manifest['sources']:
            entry.update(status=EXPERIMENTAL_STATUS,geometry_qualified=False,experimental_export=True,
                reasons=['changed_near_ground_corner_support_clearance_over_1cm'])
        path.write_text(json.dumps(manifest))
        fake_original_reset(task,manifest,task.ids)
        identity=json.loads((folder/'out/reference_goal_contract.json').read_text())
        assert identity['manifest_verbatim']==manifest and identity['geometry_qualified'] is False
        assert identity['experimental_export'] is True
        logs=[json.loads(line) for line in (folder/'out/reference_goal_resets.jsonl').read_text().splitlines()]
        assert logs[0]['experimental_export'] is True and logs[0]['geometry_qualified'] is False
        manifest['experimental_export']=False
        try:validate_plan_status(manifest)
        except ValueError as error:assert 'explicit' in str(error)
        else:raise AssertionError('Unmarked experimental plan accepted')
        folder=root/'prepare';folder.mkdir();task,manifest,path=fixture(folder,single=True)
        cfg=dict(task.cfg,motion_bank=manifest['original_bank'],tracker_checkpoint='original_flat_checkpoint',
                 scene_specs=[dict(scene_dir=str(folder/'scene'),forbidden_id=3)])
        source=folder/'source.json';source.write_text(json.dumps(cfg))
        checkpoint=folder/'teacher50.pt'
        torch.save(dict(policy_kind=POLICY_KIND,feature_version=FEATURE_VERSION,update=50,
            config=dict(stop_reference_version='original_freeze_v5_dynamic',surface_mjcf_sha256='a'*64)),checkpoint)
        args=NS(source_config=str(source),manifest=str(path),output=str(folder/'evaluation'),checkpoint=str(checkpoint),
                original_bank=None,planned_bank=None,record=False,record_env=0)
        with contextlib.redirect_stdout(io.StringIO()):prepared,provenance=prepare(args)
        assert len(prepared['cases'])==24 and provenance['num_envs']==24
        assert prepared['motion_bank']==manifest['planned_bank']
        assert prepared['stop_reference_version']==cfg['stop_reference_version']
        assert json.loads(source.read_text())==cfg
    text=Path(__file__).with_name('v5_task.py').read_text()
    reset=text[text.index('    def reset('):text.index('    def features(')]
    assert reset.index('self.deadline[ids]=')<reset.index('apply_original_goal(self,ids,mids)')<reset.index('self.prev_distance[ids]=')<reset.index('feature=self.features()')
    print(json.dumps(dict(original_goal_before_history_and_prev_distance=True,xyz_origin_and_room_added_once=True,
        planned_goal_recorded_separately=True,source_frames_and_deadline_unchanged=True,
        partial_reset_preserves_other_envs=True,evaluation_collection_training_share_hook=True,
        no_random_draws=True,rejected_plan_and_wrong_deadline_refused=True,
        identity_goal_and_history_bitwise_unchanged=True,float32_reassociation_fixture_exercised=True,
        changed_reference_goal_matches_original_runtime_bitwise=True,
        single_source_prepare_preserves_original_config=True,fixed_24_slots=True,simulator_runs=0),indent=2))


if __name__=='__main__':main()
