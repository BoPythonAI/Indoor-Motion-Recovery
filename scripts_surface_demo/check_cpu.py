"""Synthetic CPU interface checks only: no model inference, simulator or subprocess."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import run_system as demo


def expect_rejected(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid binding was accepted')


def check():
    root=Path(__file__).resolve().parent.parent
    compiler=root/('scripts_demo' if (root/'scripts_demo/task_program.py').is_file() else 'scripts')
    validator=demo.module(compiler/'task_program.py','demo_test_original_validator')
    meta=dict(scene_id='synthetic_scene',target_object_id=10,objects=[{'object_id':10},{'object_id':17},{'object_id':18}])
    program=dict(schema_version='task_program_v1',status='COMPILED',instruction=demo.INSTRUCTION,
        scene_id=meta['scene_id'],execution_gate='PERMIT_DEVELOPMENT_ONLY',
        goal={'instance_id_from_proposal_map':10},forbidden_instance_id=17,
        contact_events=[dict(mode='forbidden',object_instance_id_from_proposal_map=17,
                             body_parts=['all'],phase=['walking','recovery'])],
        source={'model':'Qwen3-VL-4B-Instruct'},
        parsed_relations=dict(goal={'relation':'farthest'},forbidden={'relation':'nearest'}))
    assert demo.bind_program(program,meta,validator)==dict(goal_instance_id=10,forbidden_instance_id=17)
    for key,value in [('status','ABSTAIN'),('instruction','different request'),('forbidden_instance_id',18)]:
        bad=copy.deepcopy(program);bad[key]=value
        expect_rejected(lambda:demo.bind_program(bad,meta,validator))
    for path,value in [(('goal','instance_id_from_proposal_map'),17),
                       (('parsed_relations','goal','relation'),'nearest'),
                       (('contact_events',0,'body_parts'),['feet']),
                       (('contact_events',0,'phase'),['walking'])]:
        bad=copy.deepcopy(program);part=bad
        for key in path[:-1]:part=part[key]
        part[path[-1]]=value
        expect_rejected(lambda:demo.bind_program(bad,meta,validator))
    with tempfile.TemporaryDirectory(prefix='surface_demo_cpu_') as directory:
        tmp=Path(directory);scene=tmp/'scene';scene.mkdir()
        demo.write(scene/'scene_metadata.json',meta)
        bank=tmp/'015317.pt';bank.write_bytes(b'Synthetic bank identity only')
        metadata=tmp/'015317.json';demo.write(metadata,[dict(motion_id='015317',scene_dir=str(scene))])
        flat=tmp/'flat.pt';flat.write_bytes(b'Synthetic flat identity only')
        terrain=tmp/'terrain.pt';terrain.write_bytes(b'Synthetic terrain identity only')
        mjcf=tmp/'robot.xml';mjcf.write_bytes(b'Synthetic MJCF identity only')
        picture=tmp/'annotated.png';picture.write_bytes(b'Synthetic image existence fixture only')
        manifest=tmp/'manifest.json';demo.write(manifest,[dict(scene_id=meta['scene_id'],annotated_images=[str(picture)])])
        import hashlib
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        prior=dict(flat_checkpoint=str(flat),terrain_checkpoint=str(terrain),flat_sha256=sha(flat),terrain_sha256=sha(terrain))
        feature='v3_terrain_reference__terrain_correction_v1__foot_clearance_v1'
        geometry='mjcf_foot_obb_forbidden_aabb_sat15_v1'
        state=dict(policy_kind='terrain_prior_surface_correction_v1',feature_version=feature,
            geometry_version=geometry,surface_mjcf_path=str(mjcf),surface_mjcf_sha256=sha(mjcf),
            frozen_prior=prior,stop_reference_version='original_freeze_v5_dynamic')
        cfg=dict(motion_bank=str(bank),motion_metadata=str(metadata),
            stop_reference_version='original_freeze_v5_dynamic',feature_version=feature,
            tracker_checkpoint=str(flat),terrain_checkpoint=str(terrain),
            surface_mjcf_path=str(mjcf),scene_specs=[dict(scene_dir=str(scene),forbidden_id=17,fixture_extra='preserved')])
        config=tmp/'scene_config.json';demo.write(config,cfg)
        checkpoint=tmp/'checkpoint.pt'
        args=argparse.Namespace(root=str(root),teacher_runtime=None,student_runtime=None,
            compiler_directory=str(compiler),scene_config=str(config),manifest=str(manifest),
            checkpoint=str(checkpoint),method='teacher',execute_steps=1,record_env=None,
            output=str(tmp/'prepared_teacher'),prepare_only=True)
        for kind,execute in [('teacher',1),('bc',1),('diffusion',2)]:
            selected=copy.deepcopy(state)
            if kind!='teacher':
                selected.update(policy_kind='terrain_prior_surface_correction_student_v1',kind=kind,
                    action_target_version='executed_total_minus_composed_terrain_prior_v1',
                    teacher_provenance={'kind':'terrain_prior_surface_correction_v1'},
                    teacher_identity='synthetic',teacher_checkpoint_sha256='a'*64,
                    motion_bank_sha256='b'*64,motion_metadata_sha256='c'*64,
                    horizon=6,ema_model={'synthetic':torch.zeros(1)})
            torch.save(selected,checkpoint)
            args.method=kind;args.execute_steps=execute;args.output=str(tmp/('prepared_'+kind))
            with patch.object(demo.subprocess,'run',side_effect=AssertionError('Prepare mode launched a process')):
                with contextlib.redirect_stdout(io.StringIO()):plan=demo.run(args)
            assert plan['status']=='PREPARED_NOT_EXECUTED'
            assert plan['num_envs']==24 and len(plan['cases'])==24
            assert plan['compiled_program'] is None and plan['evaluation'] is None
            assert 'scripts_surface' not in plan['parser_command']
            assert '--instruction' in plan['parser_command'] and 'vlm_relations.py' in plan['parser_command']
            assert ('scripts_teacher_surface' if kind=='teacher' else 'scripts_student_surface') in plan['physics_command']
            assert plan['runtime']['teacher_runtime']==str((root/'scripts_teacher_surface').resolve())
            assert plan['runtime']['task_sources']['v5_task.py']==sha(root/'scripts_teacher_surface/v5_task.py')
            assert plan['planned_reference'] is None
            assert not (Path(args.output)/'run_config.json').exists()
        # Exercise orchestration with explicitly fake parser/physics outputs. No subprocess runs.
        args.prepare_only=False;args.output=str(tmp/'synthetic_orchestration');args.record_env=16
        calls=[]
        def fake_subprocess(command,**kwargs):
            calls.append(command)
            output=Path(args.output)
            if len(calls)==1:
                (output/'qwen').mkdir()
                demo.write(output/'qwen/task_program.json',program)
                demo.write(output/'qwen/results.json',[dict(id='task',raw='SYNTHETIC CPU FIXTURE',program=program)])
            else:
                actual=demo.read(output/'run_config.json')
                assert actual['scene_specs'][0]['fixture_extra']=='preserved'
                assert actual['forbidden_id']==actual['scene_specs'][0]['forbidden_id']==17
                assert actual['task_binding']['goal_instance_id']==10
                if args.method=='teacher':assert actual['checkpoint']==str(checkpoint)
                else:assert actual['student_weights']=='ema' and actual['execute_steps']==2
                assert actual['record_env']==16 and actual['record']
                if args.teacher_runtime:
                    assert command[1]=='SURFACE_TEACHER_SCRIPTS='+str(Path(args.teacher_runtime).resolve())
                    for key in plan_fields:assert actual[key]==plan_fields[key],key
                    assert actual['demo_runtime']['teacher_runtime']==str(Path(args.teacher_runtime).resolve())
                (output/'physics').mkdir()
                demo.write(output/'physics/evaluation.json',dict(episodes=[dict(c,success=False) for c in actual['cases']],summary={}))
                demo.write(output/'physics/trace.json',{'synthetic_fixture':True})
        with patch.object(demo.subprocess,'run',side_effect=fake_subprocess):
            with contextlib.redirect_stdout(io.StringIO()):result=demo.run(args)
        assert len(calls)==2 and result['selected_episode']['bw']==.75
        assert result['selected_episode']['success'] is False
        # All three controllers use the same explicitly selected planned task.
        original=tmp/'original_015317.pt';original.write_bytes(b'Synthetic original bank identity')
        reference_plan=tmp/'reference_plan.json'
        reference_manifest=dict(plan_version='goal_preserving_xy_reference_shift_v1',
            status='PLANNED_PENDING_PHYSICS',original_bank=str(original),original_bank_sha256=sha(original),
            planned_bank_sha256=sha(bank),original_metadata_sha256=sha(metadata),mjcf_sha256=sha(mjcf),
            sources=[dict(source_id='015317',status='PLANNED_PENDING_PHYSICS')])
        demo.write(reference_plan,reference_manifest)
        plan_fields=dict(reference_planning_version=reference_manifest['plan_version'],
            reference_plan_manifest=str(reference_plan),reference_plan_manifest_sha256=sha(reference_plan),
            reference_original_bank=str(original),original_goal_world_in_scene=[[1.,2.,.8]],
            planned_reference_goal_world_in_scene=[[1.,1.92,.8]])
        demo.write(config,dict(cfg,**plan_fields))
        args.teacher_runtime=str(root/'scripts_teacher_planned')
        for kind in ('teacher','bc','diffusion'):
            if kind=='teacher':selected=copy.deepcopy(state)
            else:
                selected=copy.deepcopy(state)
                selected.update(policy_kind='terrain_prior_surface_correction_student_v1',kind=kind,
                    action_target_version='executed_total_minus_composed_terrain_prior_v1',
                    teacher_provenance={'kind':'terrain_prior_surface_correction_v1'},teacher_identity='synthetic',
                    teacher_checkpoint_sha256='a'*64,motion_bank_sha256='b'*64,motion_metadata_sha256='c'*64,
                    horizon=6,ema_model={'synthetic':torch.zeros(1)})
            torch.save(selected,checkpoint)
            args.method=kind;args.execute_steps=1 if kind=='teacher' else 2
            args.prepare_only=True;args.output=str(tmp/('planned_prepare_'+kind))
            with patch.object(demo.subprocess,'run',side_effect=AssertionError('Prepare mode launched a process')):
                with contextlib.redirect_stdout(io.StringIO()):prepared=demo.run(args)
            assert prepared['runtime']['teacher_runtime']==str((root/'scripts_teacher_planned').resolve())
            assert prepared['runtime']['task_sources']['reference_goal_contract.py']==sha(root/'scripts_teacher_planned/reference_goal_contract.py')
            assert prepared['planned_reference']['reference_plan_manifest_sha256']==sha(reference_plan)
            assert 'SURFACE_TEACHER_SCRIPTS=' in prepared['physics_command']
            args.prepare_only=False;args.output=str(tmp/('planned_fake_execution_'+kind));calls.clear()
            with patch.object(demo.subprocess,'run',side_effect=fake_subprocess):
                with contextlib.redirect_stdout(io.StringIO()):result=demo.run(args)
            assert len(calls)==2 and result['goal_instance_id']==10 and result['forbidden_instance_id']==17
            assert result['runtime']==prepared['runtime']
        # A default surface runtime must not silently execute a planned bank with a shifted goal.
        args.teacher_runtime=None
        expect_rejected(lambda:demo.preflight(args))
        args.teacher_runtime=str(root/'scripts_teacher_planned')
        for kind in ('bc','diffusion'):
            selected.update(kind=kind,horizon=1)
            torch.save(selected,checkpoint)
            args.method=kind;args.execute_steps=1;args.prepare_only=True
            args.output=str(tmp/('planned_h1_'+kind))
            with patch.object(demo.subprocess,'run',side_effect=AssertionError('H1 prepare launched process')):
                with contextlib.redirect_stdout(io.StringIO()):h1=demo.run(args)
            assert h1['method']==kind and h1['controller']['trained_horizon']==1
            assert h1['controller']['execute_steps']==1 and h1['controller']['inference_seed']==17
            args.execute_steps=2
            expect_rejected(lambda:demo.preflight(args))
        args.execute_steps=1
        # A wrong model/reference must fail before even writing a plan.
        selected['frozen_prior']['terrain_sha256']='d'*64;torch.save(selected,checkpoint)
        args.prepare_only=True;args.output=str(tmp/'rejected')
        expect_rejected(lambda:demo.preflight(args))
        assert not Path(args.output).exists()
    print('SURFACE_DEMO_CPU_PASSED: original_validator; strict_goal10_forbid17; exact_instruction; '
          'teacher_BC_diffusion; explicit_EMA; prepare_no_subprocess; 24_cases; '
          'synthetic_orchestration_only; selected_failure_preserved; prior_mismatch_rejected; '
          'planned_shared_task_all_controllers; runtime_SHA_evidence; original_goal_fields_preserved')


if __name__=='__main__':check()
