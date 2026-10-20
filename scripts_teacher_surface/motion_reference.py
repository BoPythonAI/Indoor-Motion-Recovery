"""Explicit motion/reference provenance; never infer a bank from metadata name."""
import hashlib
from pathlib import Path


REFERENCE_KEYS = ('motion_metadata', 'motion_metadata_sha256', 'motion_bank',
                  'motion_bank_sha256', 'stop_reference_version')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_motion_reference(config, motion_bank=None, require_version=True):
    selected = config.get('motion_bank')
    if selected and motion_bank and Path(selected).resolve() != Path(motion_bank).resolve():
        raise ValueError('Actual/override motion bank differs from the selected teacher motion_bank')
    selected = selected or motion_bank
    if not selected:
        raise ValueError('Explicit motion_bank is required; metadata filenames do not identify the bank')
    if not config.get('motion_metadata'):
        raise ValueError('Selected teacher must specify actual motion_metadata')
    version = config.get('stop_reference_version')
    if not isinstance(version, str) or not version.strip():
        if require_version:
            raise ValueError('Selected teacher must specify stop_reference_version (including unchanged-source references)')
        version = 'unversioned_reference'
    if require_version and version == 'unversioned_reference':
        raise ValueError('Explicit reference version is required before preparing new student data')
    result = dict(stop_reference_version=version)
    for key, filename in [('motion_bank', selected), ('motion_metadata', config['motion_metadata'])]:
        path = Path(filename).resolve()
        if not path.is_file():
            raise ValueError(f'Selected {key} file does not exist: {path}')
        actual = file_sha256(path)
        expected = config.get(key + '_sha256')
        if expected and expected != actual:
            raise ValueError(f'Selected {key} SHA256 does not match actual file bytes')
        result[key], result[key + '_sha256'] = str(path), actual
    return result


def reference_from_provenance(provenance):
    """Read recorded fields without reopening training files at deployment."""
    return {key: provenance[key] for key in REFERENCE_KEYS if key in provenance}


def actual_motion_argument(argv):
    values = []
    for index, arg in enumerate(argv):
        if arg == '--motion-file':
            if index + 1 >= len(argv):
                raise ValueError('--motion-file needs its explicit path')
            values.append(argv[index + 1])
        elif arg.startswith('--motion-file='):
            values.append(arg.split('=', 1)[1])
    if len(values) != 1:
        raise ValueError('Runtime must receive exactly one explicit --motion-file')
    return str(Path(values[0]).resolve())
