"""Use the selected teacher runtime unchanged; no copies of scene/task code."""
import os
import sys
from pathlib import Path


def install():
    own = Path(__file__).resolve().parent
    root = Path(os.environ.get('SURFACE_TEACHER_SCRIPTS', own.parent / 'scripts_teacher_surface')).resolve()
    required = ('warm_task.py', 'warm_prior.py', 'teacher_factory.py', 'v5_task.py',
                'recovery_runtime.py', 'scene_adapter.py', 'scene_heightmap.py', 'collect_recovery.py',
                'surface_contract.py', 'foot_clearance.py', 'warm_task_base.py', 'surface_curriculum.py')
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Surface teacher runtime missing at {root}: {missing}')
    for path in (str(own), str(root)):
        if path in sys.path:
            sys.path.remove(path)
    sys.path[:0] = [str(own), str(root)]
    return root
