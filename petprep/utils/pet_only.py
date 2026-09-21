"""Resource and image preparation for direct PET normalization."""

from pathlib import Path

import nibabel as nb
import numpy as np


def collect_pet_only_derivatives(pet_file, derivatives):
    """Find a motion reference/transform pair using PETPrep's derivative queries.

    Keep the pair from one derivative dataset. An unrelated or partial dataset
    must not replace a complete pair, and acquisitions without optional BIDS
    entities must not match derivatives belonging to a different acquisition.
    """
    from petprep.utils.bids import collect_derivatives, extract_entities

    entities = extract_entities(pet_file)
    for entity in (
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
    ):
        entities.setdefault(entity, None)
    complete = {}
    incomplete = {}
    for directory in (derivatives or {}).values():
        cache = collect_derivatives(derivatives_dir=directory, entities=entities)
        reference = cache.get('hmc_petref')
        transforms = cache.get('transforms', {}).get('hmc')
        pair = {'hmc_petref': reference, 'transforms': {'hmc': transforms}}
        if reference and transforms:
            complete = pair
        elif reference or transforms:
            incomplete = pair
    return complete or incomplete


def select_motion_derivatives(precomputed, nvols):
    """Validate cached motion outputs before omitting estimation from the graph."""
    import nitransforms as nt

    def single(value, label):
        if isinstance(value, (list, tuple)):
            if len(value) > 1:
                raise ValueError(f'Ambiguous precomputed {label}: {value}')
            value = value[0] if value else None
        return value

    reference = single(precomputed.get('hmc_petref'), 'HMC reference')
    transforms = single(precomputed.get('transforms', {}).get('hmc'), 'HMC transforms')
    if (reference is None) != (transforms is None):
        raise ValueError(
            'PET-only motion reuse requires both desc-hmc_petref and '
            'from-orig_to-petref motion transforms from the same derivative dataset.'
        )
    if reference is None:
        return None, None
    image = nb.load(reference)
    if image.ndim != 3:
        raise ValueError(f'The precomputed HMC reference must be 3D: {reference}')
    matrices = np.asarray(nt.linear.load(transforms, fmt='itk').matrix)
    # A single affine is (4, 4); a motion mapping is (nframes, 4, 4).
    count = 1 if matrices.ndim == 2 else matrices.shape[0]
    if count != nvols:
        raise ValueError(
            f'Precomputed HMC transform count ({count}) does not match '
            f'the number of PET frames ({nvols}): {transforms}'
        )
    if not np.isfinite(matrices).all():
        raise ValueError(f'Precomputed HMC transforms must be finite: {transforms}')
    return str(reference), str(transforms)


def template_resources(template, specification):
    """Describe the fixed registration images independently of any segmentation."""
    spec = dict(specification)
    resolution = spec.get('res', 1)
    resolution = 1 if resolution == 'native' else int(resolution)
    common = {'resolution': resolution, 'cohort': spec.get('cohort'), 'space': None}
    queries = {
        'template': {
            **common,
            'suffix': 'T1w',
            'desc': None,
            'atlas': None,
            'extension': '.nii.gz',
        },
        'mask': {
            **common,
            'suffix': 'mask',
            'desc': 'brain',
            'atlas': None,
            'extension': '.nii.gz',
        },
    }
    resources = {
        name: {'source': 'templateflow', 'template': template, 'query': query}
        for name, query in queries.items()
    }
    return resources


def resolve_resources(template, specification, segmentation):
    """Resolve TemplateFlow images in the output space and their atlas label table."""
    import hashlib

    import pandas as pd

    from petprep.utils.atlas import _resolve_resource, templateflow_atlas_resources
    from petprep.utils.pet_only import template_resources

    spec = dict(specification)
    resources = template_resources(template, spec)
    resources.update(templateflow_atlas_resources(segmentation, template, spec))
    queries = {
        name: {**resource['query'], 'template': resource['template']}
        for name, resource in resources.items()
    }
    paths = {}
    for name, resource in resources.items():
        try:
            paths[name] = _resolve_resource(resource['template'], resource)
        except ValueError as exc:
            raise ValueError(
                f'PET-only {name} unavailable for space {template}, --seg {segmentation}. '
                f'TemplateFlow query: {queries[name]}. Install or publish the matching '
                f'resource in TemplateFlow; PET-only mode does not estimate subject-specific '
                f'segmentations or use label tables from outside TemplateFlow. {exc}'
            ) from exc

    for name in ('template', 'mask', 'segmentation'):
        img = nb.load(paths[name])
        data = img.get_fdata()
        if img.ndim != 3 or not np.isfinite(data).all() or not np.any(data):
            raise ValueError(f'{name} must be a finite, nonempty 3D image.')
        if name == 'segmentation' and not np.allclose(data, np.rint(data)):
            raise ValueError('The TemplateFlow atlas must contain discrete integer labels.')
    fixed, mask = (nb.load(paths[key]) for key in ('template', 'mask'))
    if fixed.shape != mask.shape or not np.allclose(fixed.affine, mask.affine):
        raise ValueError('The template and its brain mask must share a grid.')
    labels = pd.read_csv(paths['labels'], sep='\t')
    if not {'index', 'name'}.issubset(labels.columns):
        raise ValueError('TemplateFlow label tables must contain index and name columns.')
    indices = pd.to_numeric(labels['index'], errors='coerce').to_numpy()
    if (
        not np.isfinite(indices).all()
        or not np.equal(indices, np.rint(indices)).all()
        or labels['name'].isna().any()
    ):
        raise ValueError('Atlas label indices must be integers and names must be present.')
    if {'frame_start', 'frame_end'} & set(labels['name']):
        raise ValueError('Atlas names must not conflict with TAC timing column names.')
    if labels['index'].duplicated().any() or labels['name'].duplicated().any():
        raise ValueError('Atlas label indices and names must be unique.')
    unknown = (
        set(np.unique(nb.load(paths['segmentation']).get_fdata())) - set(labels['index']) - {0}
    )
    if unknown:
        raise ValueError(f'Atlas labels missing from its table: {sorted(unknown)}')
    provenance = {
        'Template': template,
        'RegistrationContrast': 'T1w',
        'Atlas': segmentation,
        'SegmentationSource': 'TemplateFlow',
        'Resources': {
            name: {
                'Source': 'TemplateFlow',
                'Query': queries[name],
                'Filename': Path(path).name,
                'SHA256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }
    return paths['template'], paths['mask'], paths['segmentation'], paths['labels'], provenance


def prepare_reference(pet_file, motion_reference, metadata, strategy):
    """Use PETPrep reference strategies on the motion-corrected series."""
    from petprep.workflows.pet.fit import (
        _extract_first5min_image,
        _extract_sum_image,
        _extract_twa_image,
    )

    if strategy == 'template':
        result = motion_reference
        image = nb.load(result)
        if image.ndim == 3:
            return result
        # Static PET is sometimes encoded as a singleton 4D series.
        if image.shape[3] == 1:
            out = Path.cwd() / 'petref.nii.gz'
            nb.Nifti1Image(image.get_fdata()[..., 0], image.affine).to_filename(out)
            return str(out)
        strategy = 'twa'  # HMC disabled: no within-run motion template exists.
    kwargs = {'pet_file': pet_file, 'output_dir': Path.cwd()}
    if strategy == 'sum':
        result = _extract_sum_image(**kwargs)
    else:
        kwargs.update(
            frame_start_times=metadata.get('FrameTimesStart'),
            frame_durations=metadata.get('FrameDuration'),
        )
        if strategy == 'first5min':
            result = _extract_first5min_image(**kwargs)
        elif strategy == 'twa':
            result = _extract_twa_image(**kwargs)
        else:
            raise ValueError(f'Unsupported PET-only reference strategy: {strategy}')
    image = nb.load(result)
    if image.ndim == 4 and image.shape[3] == 1:
        out = Path.cwd() / 'petref.nii.gz'
        nb.Nifti1Image(image.get_fdata()[..., 0], image.affine).to_filename(out)
        return str(out)
    return result


def identity_motion(pet_file):
    from petprep.workflows.pet.fit import _write_identity_xforms

    image = nb.load(pet_file)
    count = image.shape[3] if image.ndim == 4 else 1
    return str(_write_identity_xforms(count, Path.cwd() / 'motion.txt'))


def write_metadata(metadata):
    import json

    out = Path.cwd() / 'metadata.json'
    out.write_text(json.dumps(metadata))
    return str(out)


def read_metadata(in_file):
    import json

    return json.loads(Path(in_file).read_text())


def sampling_support(pet_file):
    """Separate measured image support from PET intensity, including genuine zeros."""
    image = nb.load(pet_file)
    out = Path.cwd() / 'support.nii.gz'
    nb.Nifti1Image(np.isfinite(image.get_fdata()).astype('uint8'), image.affine).to_filename(out)
    return str(out)


def template_segmentation(segmentation, reference, label_table, space):
    """Sample discrete atlas labels on the output grid and reuse atlas morphometry."""
    from nibabel.processing import resample_from_to

    from petprep.utils.segmentation import atlas_segmentation_to_morph

    target = nb.load(reference)
    source = nb.load(segmentation)
    sampled = resample_from_to(source, (target.shape[:3], target.affine), order=0)
    out = Path.cwd() / 'template_dseg.nii.gz'
    nb.Nifti1Image(np.rint(sampled.get_fdata()).astype('int32'), target.affine).to_filename(out)
    morph, meta = atlas_segmentation_to_morph(str(out), label_table)
    meta.update(
        MeasurementSpace=space,
        SubjectSpecific=False,
        Description='Template atlas region volumes on the PET output grid; '
        'these are not individual anatomical volume estimates.',
    )
    return str(out), morph, meta
