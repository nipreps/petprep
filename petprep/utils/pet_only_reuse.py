"""Reuse PET-only derivatives by filename and ordinary JSON sidecar provenance."""

import json
from pathlib import Path

import nibabel as nb


def reference_metadata(metadata, strategy, hmc_off):
    """Settings accompanying a strategy-labelled registration reference."""
    return {
        'PETReferenceStrategy': strategy,
        'MotionCorrection': not hmc_off,
        **{key: metadata.get(key) for key in ('FrameTimesStart', 'FrameDuration')},
    }


def registration_metadata(petref, fixed_image, fixed_mask, reference_settings, space, res, sloppy):
    """Record which reference and TemplateFlow images were used to fit a transform."""
    import json
    from pathlib import Path

    from petprep.utils.pet_registration import REGISTRATION_MASK_DILATION_MM
    from petprep.workflows.pet.normalization import registration_parameters

    return {
        **reference_settings,
        'Sources': [Path(path).name for path in (petref, fixed_image, fixed_mask)],
        'SpatialReference': space,
        'TargetResolution': str(res),
        'RegistrationParameters': json.loads(json.dumps(registration_parameters(sloppy))),
        'RegistrationMask': {
            'Source': Path(fixed_mask).name,
            'DilationRadius': REGISTRATION_MASK_DILATION_MM,
            'DilationRadiusUnits': 'mm',
            'DilationMethod': 'Euclidean',
            'Stages': ['SyN'],
        },
    }


def resampling_metadata(pet_file, transform, registration, metadata):
    """Keep the reference provenance when linking normalized PET to its transform."""
    from pathlib import Path

    return {
        **metadata,
        **registration,
        'Sources': [Path(pet_file).name, Path(transform).name],
        'RegistrationSources': registration['Sources'],
        'RegistrationMethod': 'ANTs rigid + affine + SyN',
    }


def _matches_sidecar(path, expected):
    if not path.is_file():
        return False
    try:
        sidecar = path.with_name(path.name.split('.', 1)[0] + '.json')
        metadata = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(metadata, dict) and all(
        key in metadata and metadata[key] == value for key, value in expected.items()
    )


def _candidates(pet_file, derivatives, **query):
    """Find files afresh, including absent acquisition entities in the match."""
    from bids.layout import parse_file_entities

    acquisition = (
        'subject',
        'session',
        'task',
        'acquisition',
        'ceagent',
        'tracer',
        'reconstruction',
        'run',
        'direction',
        'echo',
        'part',
    )
    source = parse_file_entities(str(Path(pet_file).absolute()))
    expected = {key: source.get(key) for key in acquisition}
    expected.update(query)
    for root in reversed(list((derivatives or {}).values())):
        directory = Path(root) / f'sub-{source["subject"]}'
        if source.get('session'):
            directory /= f'ses-{source["session"]}'
        for path in sorted((directory / 'pet').glob(f'*_{query["suffix"]}{query["extension"]}')):
            entities = parse_file_entities(str(path), config=['bids', 'derivatives'])
            if all(entities.get(key) == value for key, value in expected.items()):
                yield path


def find_reference(pet_file, metadata, precomputed, derivatives, strategy, hmc_off):
    """Reuse desc-<petref> for this acquisition when reference settings agree."""
    from petprep.utils.pet_only import select_motion_derivatives

    image = nb.load(pet_file)
    nframes = image.shape[3] if image.ndim == 4 else 1
    if nframes > 1 and not hmc_off:
        _, motion = select_motion_derivatives(precomputed or {}, nframes)
        if motion is None:
            return None  # A newly fitted motion model needs a new registration reference.
    expected = reference_metadata(metadata, strategy, hmc_off)
    for path in _candidates(
        pet_file,
        derivatives,
        desc=strategy,
        suffix='petref',
        extension='.nii.gz',
        space=None,
        res=None,
        cohort=None,
    ):
        if _matches_sidecar(path, expected):
            return str(path)
    return None


def find_target_derivatives(
    pet_file,
    metadata,
    petref,
    derivatives,
    template,
    spec,
    strategy,
    hmc_off,
    sloppy,
):
    """Match transform sources, then normalized PET/support sources, per target."""
    if petref is None:
        return None, None
    from petprep.utils.atlas import _resolve_resource
    from petprep.utils.pet_only import template_resources

    space = template + (f'+{spec["cohort"]}' if 'cohort' in spec else '')
    res = str(spec.get('res', 1))
    candidates = list(
        _candidates(
            pet_file,
            derivatives,
            suffix='xfm',
            extension='.h5',
            res=res,
            desc=None,
            **{'from': 'petref', 'to': space, 'mode': 'image'},
        )
    )
    if not candidates:
        return None, None
    resources = template_resources(template, spec)
    fixed = _resolve_resource(template, resources['template'])
    mask = _resolve_resource(template, resources['mask'])
    expected = registration_metadata(
        petref,
        fixed,
        mask,
        reference_metadata(metadata, strategy, hmc_off),
        space,
        res,
        sloppy,
    )
    for forward in candidates:
        # The pair and warped reference must be in the same derivative directory.
        stem = forward.name.split('_res-', 1)[0]
        inverse = forward.with_name(
            forward.name.replace(f'_from-petref_to-{space}_', f'_from-{space}_to-petref_')
        )
        spatial = f'_space-{space}' + (f'_res-{res}' if res != 'native' else '')
        warped = forward.with_name(f'{stem}{spatial}_desc-{strategy}_petref.nii.gz')
        if not all(_matches_sidecar(path, expected) for path in (forward, inverse, warped)):
            continue
        registration = {'forward': str(forward), 'inverse': str(inverse), 'warped': str(warped)}
        pet = forward.with_name(f'{stem}{spatial}_desc-preproc_pet.nii.gz')
        support = forward.with_name(f'{stem}{spatial}_desc-support_probseg.nii.gz')
        expected_pet = resampling_metadata(pet_file, forward, expected, metadata)
        if all(_matches_sidecar(path, expected_pet) for path in (pet, support)):
            return registration, {'pet': str(pet), 'support': str(support)}
        return registration, None
    return None, None
