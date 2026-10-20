"""Explicit contracts: these students never consume legacy flat residual labels."""
import hashlib
import json

FEATURE_VERSION = 'v3_terrain_reference__terrain_correction_v1__foot_clearance_v1'
ACTION_TARGET_VERSION = 'executed_total_minus_composed_terrain_prior_v1'
POLICY_KIND = 'terrain_prior_surface_correction_student_v1'
TEACHER_KIND = 'terrain_prior_surface_correction_v1'
GEOMETRY_VERSION = 'mjcf_foot_obb_forbidden_aabb_sat15_v1'
GEOMETRY_KEYS = ('geometry_version','surface_mjcf_sha256')
REFERENCE_KEYS = ('stop_reference_version', 'motion_bank_sha256', 'motion_metadata_sha256')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Missing or invalid {name}; explicit checkpoint/bank provenance is required')
    return value


def provenance_from_episode(data):
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Wrong features: require surface terrain/foot-clearance correction feature version')
    if data.get('action_target_version') != ACTION_TARGET_VERSION or 'executed_correction' not in data:
        raise ValueError('Legacy/full residual data rejected: explicit executed_correction and target version required')
    provenance = data.get('teacher_provenance') or {}
    if provenance.get('kind') != TEACHER_KIND:
        raise ValueError('A surface correction teacher provenance is required')
    geometry={key:data.get(key,provenance.get(key)) for key in GEOMETRY_KEYS}
    if geometry['geometry_version']!=GEOMETRY_VERSION:
        raise ValueError('Surface foot-clearance geometry version is required')
    _sha(geometry['surface_mjcf_sha256'],'surface_mjcf_sha256')
    prior = data.get('frozen_prior') or {k: provenance.get(k) for k in
            ('flat_checkpoint', 'flat_sha256', 'terrain_checkpoint', 'terrain_sha256')}
    for key in ('flat_sha256', 'terrain_sha256'):
        _sha(prior.get(key), key)
    teacher_sha = data.get('teacher_checkpoint_sha256') or provenance.get('teacher_checkpoint_sha256')
    _sha(teacher_sha, 'teacher_checkpoint_sha256')
    reference = {key: data.get(key, provenance.get(key)) for key in REFERENCE_KEYS}
    if not reference['stop_reference_version']:
        raise ValueError('An explicit motion reference version is required')
    for key in REFERENCE_KEYS[1:]:
        _sha(reference[key], key)
    if not data.get('teacher_identity'):
        raise ValueError('Missing teacher identity')
    return dict(feature_version=FEATURE_VERSION, action_target_version=ACTION_TARGET_VERSION,
                teacher_identity=data['teacher_identity'], teacher_checkpoint=data.get('teacher_checkpoint'),
                teacher_checkpoint_sha256=teacher_sha, frozen_prior=prior,
                teacher_provenance=provenance,
                surface_mjcf_path=data.get('surface_mjcf_path',provenance.get('surface_mjcf_path')),
                **reference,**geometry)


def signature(contract):
    # Paths are recorded, but moving byte-identical checkpoints does not change identity.
    keys = ('feature_version', 'action_target_version', 'teacher_checkpoint_sha256', *REFERENCE_KEYS, *GEOMETRY_KEYS)
    identity = {key: contract[key] for key in keys}
    identity['prior'] = {key: contract['frozen_prior'][key] for key in ('flat_sha256', 'terrain_sha256')}
    return json.dumps(identity, sort_keys=True)


def validate_checkpoint(state):
    if state.get('policy_kind') != POLICY_KIND:
        raise ValueError('Legacy student checkpoint rejected: expected correction student policy_kind')
    # The same provenance is mandatory for saved student models and their teacher data.
    provenance_from_episode(dict(state, executed_correction=True))


def validate_runtime(state, task, prior, config):
    validate_checkpoint(state)
    if task.feature_version != FEATURE_VERSION or state['obs_dim'] != task.history.flatten(1).shape[-1]:
        raise ValueError('Student features/history dimension do not match the warm task')
    for key in ('flat_sha256', 'terrain_sha256'):
        if state['frozen_prior'][key] != prior.identity[key]:
            raise ValueError(f'Frozen prior changed: {key}')
    if state['stop_reference_version'] != config.get('stop_reference_version'):
        raise ValueError('Training and deployment reference versions differ')
    for key in GEOMETRY_KEYS:
        if state[key]!=config.get(key):
            raise ValueError('Surface geometry changed between training and deployment: '+key)
    # Validation/test deliberately use different banks. Bind their hashes in run provenance,
    # never require a held-out scene to equal the training bank.
    for key in REFERENCE_KEYS[1:]:
        _sha(config.get(key), 'deployment_' + key)
    for key in ('task_owned_gate', 'momentum_observer', 'hybrid_gate'):
        if config.get(key):
            raise ValueError('Correction student has a continuous terrain prior and no observer/gate')
