"""Evaluation/collection adapter; physics, task success and reference stay in warm runtime."""
import json
import os
from pathlib import Path
import torch
from contracts import file_sha256


def run(agent):
    from warm_task import WarmRecoveryTask
    from v5_task import evaluate
    from controller import make_student
    config = json.loads(Path(os.environ['V4_CONFIG']).read_text())
    verified = os.environ.get('V4_VERIFIED_MOTION_REFERENCE')
    if not verified:
        raise ValueError('Use student entry.py to bind the actual deployment motion bank')
    config.update(json.loads(verified))
    torch.manual_seed(config.get('seed', 17)); torch.set_num_threads(4)
    agent.eval(); agent.evaluator._disable_perturbations()
    env = agent.env
    if env.robot_config.number_of_actions != 69 or env.config.action_config['action_transform'] != 'tanh':
        raise ValueError('Student requires the frozen tracker normalized 69D action contract')
    env.reset()
    task = WarmRecoveryTask(agent, config)
    task.reset()
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    if config['mode'] == 'evaluate':
        if config.get('method') not in ('bc', 'diffusion'):
            raise ValueError('Use the teacher entry for terrain_prior/terrain_warm baselines')
        controller = make_student(task, config)
        report = dict(deployment=config, student_checkpoint_sha256=file_sha256(config['student_checkpoint']),
                      student_training={key: controller.state[key] for key in
                        ('feature_version', 'action_target_version', 'teacher_checkpoint_sha256',
                         'frozen_prior', 'stop_reference_version', 'motion_bank_sha256', 'motion_metadata_sha256',
                         'geometry_version','surface_mjcf_sha256','surface_mjcf_path')})
        (output / 'student_provenance.json').write_text(json.dumps(report, indent=2))
        return evaluate(task, controller, config)
    if config['mode'] == 'collect':
        from teacher_factory import read_checkpoint, make_warm_teacher
        from student_collection import run_collection
        if not config.get('checkpoint'):
            raise ValueError('Collection needs an explicitly selected qualified warm teacher checkpoint')
        teacher = make_warm_teacher(task, config, read_checkpoint(config['checkpoint']))
        teacher.model.eval()
        return run_collection(task, teacher, config)
    raise ValueError('Use mode=evaluate/collect here; offline students train with train_student.py')
