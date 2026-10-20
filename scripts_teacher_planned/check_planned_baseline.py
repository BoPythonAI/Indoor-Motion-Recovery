"""Synthetic CPU configuration check. No policy/physics/model loading."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from motion_reference import file_sha256
from reference_goal_contract import PLAN_VERSION
from prepare_planned_baseline import prepare


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


def main():
    with tempfile.TemporaryDirectory(prefix='planned_baseline_') as directory:
        root=Path(directory)
        original=root/'original.pt';original.write_bytes(b'Original synthetic bank')
        bank=root/'planned.pt';bank.write_bytes(b'Planned synthetic bank')
        flat=root/'flat.pt';flat.write_bytes(b'Original synthetic flat tracker')
        mjcf=root/'robot.xml';mjcf.write_bytes(b'Synthetic collision MJCF')
        metadata=root/'metadata.json'
        write(metadata,[dict(motion_id='015571',source_frames=100,scene_dir=str(root/'scene'))])
        manifest=dict(plan_version=PLAN_VERSION,status='PLANNED_PENDING_PHYSICS',
            original_bank=str(original),original_bank_sha256=file_sha256(original),
            planned_bank=str(bank),planned_bank_sha256=file_sha256(bank),
            original_metadata_sha256=file_sha256(metadata),mjcf_sha256=file_sha256(mjcf),
            sources=[dict(source_id='015571',status='PLANNED_PENDING_PHYSICS')])
        plan=root/'plan.json';write(plan,manifest)
        base=dict(mode='evaluate',method='base',feature_version='v3_terrain_reference',
            motion_metadata=str(metadata),scene_specs=[dict(scene_dir=str(root/'scene'),forbidden_id=9)],
            seed=17,adaptive_phase=True,max_bw=.45,lr=.0001,
            checkpoint='unused_old_residual.pt',resume=True,student_checkpoint='unused_old_student.pt')
        # The old base is allowed to lack motion_bank/tracker_checkpoint: old launch supplied them.
        planned=dict(base,method='terrain_surface',feature_version='surface_teacher_features',
            motion_bank=str(bank),tracker_checkpoint=str(flat),terrain_checkpoint='should_never_copy.pt',
            stop_reference_version='original_freeze_v5_dynamic',
            surface_mjcf_path=str(mjcf),surface_mjcf_sha256=file_sha256(mjcf),
            geometry_version='mjcf_foot_obb_forbidden_aabb_sat15_v1',
            reference_planning_version=PLAN_VERSION,reference_plan_manifest=str(plan),
            reference_plan_manifest_sha256=file_sha256(plan),reference_original_bank=str(original),
            force_strengths=[.45,.75],surface_curriculum={'enabled':True},success_bonus=250,
            width=999,teacher_kind='terrain_surface',updates=100,lr=.001)
        source=root/'base.json';paired=root/'planned.json';write(source,base);write(paired,planned)
        args=SimpleNamespace(source_baseline_config=str(source),planned_evaluation_config=str(paired),output=str(root/'prepared'))
        with contextlib.redirect_stdout(io.StringIO()):cfg,provenance=prepare(args)
        for key in ('force_strengths','surface_curriculum','success_bonus','width','teacher_kind','updates','terrain_checkpoint'):
            assert key not in cfg,key
        for key in ('checkpoint','resume','student_checkpoint'):assert key not in cfg
        for key in ('seed','adaptive_phase','max_bw','lr','scene_specs'):
            assert cfg[key]==base[key],key
        assert cfg['feature_version']=='v3_terrain_reference' and cfg['method']=='base'
        assert cfg['motion_bank']==str(bank) and cfg['tracker_checkpoint']==str(flat)
        assert len(cfg['cases'])==24 and cfg['cases'][16]==dict(motion=0,bw=.75,direction=0,phase=.3)
        assert provenance['num_envs']==24 and provenance['execution']=='NOT_RUN'
        assert provenance['command'][-3:]==['24',str(bank),str(flat)]
        assert provenance['original_flat_checkpoint_hash_comparable'] is False
        assert provenance['manifest_verbatim']==manifest
        # Preserve source-base random-reset settings if they were present originally.
        base.update(motion_bank=str(original),tracker_checkpoint=str(flat),force_strengths=[.25,.5])
        write(source,base);args.output=str(root/'with_original_settings')
        with contextlib.redirect_stdout(io.StringIO()):cfg,provenance=prepare(args)
        assert cfg['force_strengths']==[.25,.5] and provenance['original_flat_checkpoint_hash_comparable']
        for label,update in [('scene',{'scene_specs':[dict(scene_dir=str(root/'other'),forbidden_id=9)]}),
                             ('seed',{'seed':29}),('ordering',{'cases':[dict(motion=0,bw=0,direction=0,phase=.3)]})]:
            altered=dict(planned,**update);write(paired,altered);args.output=str(root/('bad_'+label))
            try:
                with contextlib.redirect_stdout(io.StringIO()):prepare(args)
            except ValueError:pass
            else:raise AssertionError('Accepted mismatched '+label)
            assert not Path(args.output).exists()
    print('PLANNED_BASELINE_CPU_PASSED: baseline_config_preserved; no_teacher_settings_leakage; '
          'residual_checkpoints_removed; original_goal_contract; actual_flat_and_MJCF_hashes; '
          '24_slots; old_launch_only_paths_supported; mismatched_scene_seed_order_rejected; no_execution')


if __name__=='__main__':main()
