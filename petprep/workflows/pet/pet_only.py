"""Experimental PET-only preprocessing using the requested standard spaces."""

import sys

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.header import ValidateImage

from ... import config
from ...interfaces.pet_only import PETOnlyDataSink
from ...interfaces.reports import AboutSummary, SubjectSummary
from ...interfaces.resampling import ResampleSeries
from ...utils.misc import estimate_pet_mem_usage
from ...utils.pet_only import (
    collect_pet_only_derivatives,
    identity_motion,
    prepare_reference,
    read_metadata,
    resolve_resources,
    sampling_support,
    select_motion_derivatives,
    template_segmentation,
    write_metadata,
)
from ...utils.pet_only_reuse import (
    find_reference,
    find_target_derivatives,
    reference_metadata,
    registration_metadata,
    resampling_metadata,
)
from .apply import init_pet_volumetric_resample_wf
from .hmc import init_pet_hmc_wf, plan_hmc_resource_policy
from .normalization import init_pet_template_reg_wf
from .outputs import init_ds_hmc_wf, init_ds_petref_wf
from .tacs import init_pet_tacs_wf


def _function(function, inputs, outputs, name):
    return pe.Node(
        niu.Function(
            function=function,
            input_names=inputs,
            output_names=outputs,
            imports=['from pathlib import Path', 'import nibabel as nb', 'import numpy as np'],
        ),
        name=name,
    )


def _sink(source, name, **entities):
    return pe.Node(
        PETOnlyDataSink(
            base_directory=str(config.execution.petprep_dir),
            source_file=source,
            datatype='pet',
            **entities,
        ),
        name=name,
        run_without_submitting=True,
    )


def _cached_workflow(name, inputs, outputs):
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=inputs), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=list(outputs)), name='outputnode')
    for field, value in outputs.items():
        setattr(outputnode.inputs, field, value)
    workflow.add_nodes([inputnode, outputnode])
    return workflow


def init_pet_only_hmc_wf(*, pet_file, metadata, precomputed=None, name='pet_motion_wf'):
    """Stage 1: estimate or reuse PETPrep's native motion reference and transforms."""
    workflow = Workflow(name=name)
    nvols, mem_gb = estimate_pet_mem_usage(pet_file)
    threads = config.nipype.omp_nthreads
    motion_reference = motion_xfm = None
    if nvols > 1 and not config.workflow.hmc_off:
        motion_reference, motion_xfm = select_motion_derivatives(precomputed or {}, nvols)
    elif precomputed:
        config.loggers.workflow.info(
            'PET Stage 1: Ignoring precomputed motion correction for static PET or --hmc-off.'
        )
    validate = pe.Node(ValidateImage(in_file=pet_file), name='validate_pet')
    buffer = pe.Node(
        niu.IdentityInterface(fields=['motion_reference', 'motion_xfm']), name='motion_buffer'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'motion_reference', 'motion_xfm']),
        name='outputnode',
    )
    if motion_xfm:
        config.loggers.workflow.info(
            'PET Stage 1: Found head motion correction transforms and petref - skipping Stage 1'
        )
        config.loggers.workflow.info('Reusing HMC reference: %s', motion_reference)
        config.loggers.workflow.info('Reusing HMC transforms: %s', motion_xfm)
        buffer.inputs.motion_reference = motion_reference
        buffer.inputs.motion_xfm = motion_xfm
    elif nvols > 1 and not config.workflow.hmc_off:
        config.loggers.workflow.info('PET Stage 1: Adding motion correction workflow')
        if config.workflow.hmc_memory_policy == 'auto':
            policy = plan_hmc_resource_policy(
                pet_file,
                start_time=config.workflow.hmc_start_time,
                frame_durations=metadata.get('FrameDuration'),
                frame_start_times=metadata.get('FrameTimesStart'),
                fixed_frame=config.workflow.hmc_fix_frame,
            )
        else:
            policy = {
                'planned_memory_gb': mem_gb['filesize'],
                'fixed_frame': config.workflow.hmc_fix_frame,
                'subsample_threshold': None,
            }
        hmc = init_pet_hmc_wf(
            mem_gb=max(mem_gb['filesize'], policy['planned_memory_gb']),
            omp_nthreads=threads,
            fwhm=config.workflow.hmc_fwhm,
            start_time=config.workflow.hmc_start_time,
            frame_durations=metadata.get('FrameDuration'),
            frame_start_times=metadata.get('FrameTimesStart'),
            initial_frame=config.workflow.hmc_init_frame,
            fixed_frame=policy['fixed_frame'],
            subsample_threshold=policy['subsample_threshold'],
            memory_policy=config.workflow.hmc_memory_policy,
        )
        workflow.connect(
            [
                (validate, hmc, [('out_file', 'inputnode.pet_file')]),
                (
                    hmc,
                    buffer,
                    [
                        ('outputnode.petref', 'motion_reference'),
                        ('outputnode.xforms', 'motion_xfm'),
                    ],
                ),
            ]
        )
    else:
        config.loggers.workflow.info('PET Stage 1: Using identity motion transforms')
        identity = _function(identity_motion, ['pet_file'], ['transforms'], 'identity_motion')
        reference = _function(
            prepare_reference,
            ['pet_file', 'motion_reference', 'metadata', 'strategy'],
            ['petref'],
            'motion_reference',
        )
        reference.inputs.metadata = metadata
        reference.inputs.strategy = 'twa'
        workflow.connect(
            [
                (validate, identity, [('out_file', 'pet_file')]),
                (
                    validate,
                    reference,
                    [('out_file', 'pet_file'), ('out_file', 'motion_reference')],
                ),
                (reference, buffer, [('petref', 'motion_reference')]),
                (identity, buffer, [('transforms', 'motion_xfm')]),
            ]
        )

    workflow.connect(validate, 'out_file', outputnode, 'pet_file')
    if motion_xfm:
        workflow.connect(buffer, 'motion_reference', outputnode, 'motion_reference')
        workflow.connect(buffer, 'motion_xfm', outputnode, 'motion_xfm')
    if not motion_xfm:
        # Persist the SAME pair consumed by both PET-only and anatomical runs.
        ds_hmc = init_ds_hmc_wf(
            bids_root=config.execution.bids_dir, output_dir=config.execution.petprep_dir
        )
        ds_hmc.inputs.inputnode.source_files = [pet_file]
        ds_ref = init_ds_petref_wf(
            bids_root=config.execution.bids_dir,
            output_dir=config.execution.petprep_dir,
            desc='hmc',
        )
        ds_ref.inputs.inputnode.source_files = [pet_file]
        workflow.connect(
            [
                (buffer, ds_hmc, [('motion_xfm', 'inputnode.xforms')]),
                (buffer, ds_ref, [('motion_reference', 'inputnode.petref')]),
                (ds_hmc, outputnode, [('outputnode.xforms', 'motion_xfm')]),
                (ds_ref, outputnode, [('outputnode.petref', 'motion_reference')]),
            ]
        )
    return workflow


def init_pet_only_reference_wf(*, pet_file, metadata, cached=None, name='pet_reference_wf'):
    """Stage 2: construct the requested reference independently of motion fitting."""
    workflow = Workflow(name=name)
    _, mem_gb = estimate_pet_mem_usage(pet_file)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'motion_reference', 'motion_xfm']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['petref']), name='outputnode')
    if cached:
        config.loggers.workflow.info(
            'PET Stage 2: Found matching %s reference derivatives - skipping Stage 2',
            config.workflow.petref,
        )
        outputnode.inputs.petref = cached
        config.loggers.workflow.info('Reusing registration reference: %s', cached)
        workflow.add_nodes([inputnode, outputnode])
        return workflow
    config.loggers.workflow.info(
        'PET Stage 2: Building the %s registration reference',
        config.workflow.petref,
    )
    reference = _function(
        prepare_reference,
        ['pet_file', 'motion_reference', 'metadata', 'strategy'],
        ['petref'],
        'reference',
    )
    reference.inputs.metadata = metadata
    reference.inputs.strategy = config.workflow.petref
    ds_ref = init_ds_petref_wf(
        bids_root=config.execution.bids_dir,
        output_dir=config.execution.petprep_dir,
        desc=config.workflow.petref,
    )
    ds_ref.inputs.inputnode.source_files = [pet_file]
    ds_ref.get_node('ds_petref').inputs.meta_dict = reference_metadata(
        metadata, config.workflow.petref, config.workflow.hmc_off
    )
    workflow.connect(
        [
            (inputnode, reference, [('motion_reference', 'motion_reference')]),
            (reference, ds_ref, [('petref', 'inputnode.petref')]),
            (ds_ref, outputnode, [('outputnode.petref', 'petref')]),
        ]
    )
    if config.workflow.petref == 'template':
        workflow.connect(inputnode, 'pet_file', reference, 'pet_file')
    else:
        corrected = pe.Node(
            ResampleSeries(mode='constant', num_threads=config.nipype.omp_nthreads),
            name='motion_corrected',
            mem_gb=mem_gb['resampled'],
        )
        workflow.connect(
            [
                (
                    inputnode,
                    corrected,
                    [
                        ('pet_file', 'in_file'),
                        ('motion_reference', 'ref_file'),
                        ('motion_xfm', 'transforms'),
                    ],
                ),
                (corrected, reference, [('out_file', 'pet_file')]),
            ]
        )
    return workflow


def init_pet_only_prepare_wf(
    *, pet_file, metadata, precomputed=None, reference_cache=None, name='pet_prepare_wf'
):
    """Share motion fitting across targets and prepare or reuse the PET reference."""
    workflow = Workflow(name=name)
    motion = init_pet_only_hmc_wf(pet_file=pet_file, metadata=metadata, precomputed=precomputed)
    reference = init_pet_only_reference_wf(
        pet_file=pet_file, metadata=metadata, cached=reference_cache
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'petref', 'motion_xfm']),
        name='outputnode',
    )
    workflow.connect(
        [
            (
                motion,
                reference,
                [
                    ('outputnode.pet_file', 'inputnode.pet_file'),
                    ('outputnode.motion_reference', 'inputnode.motion_reference'),
                    ('outputnode.motion_xfm', 'inputnode.motion_xfm'),
                ],
            ),
            (
                motion,
                outputnode,
                [('outputnode.pet_file', 'pet_file'), ('outputnode.motion_xfm', 'motion_xfm')],
            ),
            (
                reference,
                outputnode,
                [('outputnode.petref', 'petref')],
            ),
        ]
    )
    return workflow


def init_pet_only_target_wf(
    *,
    pet_file,
    metadata,
    reference,
    name,
    registration_cache=None,
    resampling_cache=None,
):
    """Stages 3–5: normalize, resample, and extract TACs for one output space."""
    workflow = Workflow(name=name)
    template = reference.space
    spec = reference.spec
    space = template + (f'+{spec["cohort"]}' if 'cohort' in spec else '')
    resolution = spec.get('res', 1)
    threads = config.nipype.omp_nthreads
    _, mem_gb = estimate_pet_mem_usage(pet_file)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'petref', 'motion_xfm']),
        name='inputnode',
    )
    resources = _function(
        resolve_resources,
        ['template', 'specification', 'segmentation'],
        ['template', 'mask', 'segmentation', 'labels', 'provenance'],
        'resources',
    )
    resources.inputs.template = template
    resources.inputs.specification = spec
    resources.inputs.segmentation = config.workflow.seg

    # Stage 3 replaces PET-to-anatomical coregistration with direct normalization.
    if registration_cache:
        config.loggers.workflow.info(
            'PET Stage 3: Found matching registration to %s (res-%s) - skipping Stage 3',
            space,
            resolution,
        )
        artifacts = registration_cache
        registration = _cached_workflow(
            'pet_template_reg_wf',
            ['petref', 'template', 'template_mask'],
            {
                'petref2template_xfm': artifacts['forward'],
                'template2petref_xfm': artifacts['inverse'],
                'warped_petref': artifacts['warped'],
            },
        )
    else:
        config.loggers.workflow.info(
            'PET Stage 3: Adding PET-to-template registration for %s (res-%s)', space, resolution
        )
        registration = init_pet_template_reg_wf(
            omp_nthreads=threads,
            sloppy=config.execution.sloppy,
        )

    # Stage 4 applies the composed motion and normalization transforms once.
    if resampling_cache:
        config.loggers.workflow.info(
            'PET Stage 4: Found matching normalized PET and support in %s (res-%s) '
            '- skipping Stage 4',
            space,
            resolution,
        )
        artifacts = resampling_cache
        resample = _cached_workflow(
            'pet_volumetric_resample_wf',
            ['pet_file'],
            {
                'pet_file': artifacts['pet'],
                'resampling_reference': artifacts['pet'],
            },
        )
        support_resample = _cached_workflow(
            'support_resample_wf',
            ['pet_file'],
            {
                'pet_file': artifacts['support'],
            },
        )
    else:
        config.loggers.workflow.info(
            'PET Stage 4: Adding PET and sampling-support resampling to %s (res-%s)',
            space,
            resolution,
        )
        resample = init_pet_volumetric_resample_wf(
            mem_gb=mem_gb, omp_nthreads=threads, direct=True
        )
        resample.inputs.inputnode.resolution = resolution
        support_source = _function(
            sampling_support, ['pet_file'], ['out_file'], 'sampling_support'
        )
        support_resample = init_pet_volumetric_resample_wf(
            mem_gb=mem_gb, omp_nthreads=threads, direct=True, name='support_resample_wf'
        )
        support_resample.inputs.inputnode.resolution = resolution
        support_resample.get_node('resample').inputs.order = 1
        support_resample.get_node('resample').inputs.prefilter = False
        workflow.connect(
            [
                (inputnode, resample, [('pet_file', 'inputnode.pet_file')]),
                (inputnode, support_source, [('pet_file', 'pet_file')]),
                (support_source, support_resample, [('out_file', 'inputnode.pet_file')]),
            ]
        )

    # Stage 5 uses PETPrep's atlas morphometry and TAC extraction components.
    config.loggers.workflow.info(
        'PET Stage 5: Adding %s atlas morphometry and TAC extraction in %s (res-%s)',
        config.workflow.seg,
        space,
        resolution,
    )
    labels = _function(
        template_segmentation,
        ['segmentation', 'reference', 'label_table', 'space'],
        ['segmentation', 'morph', 'morph_metadata'],
        'template_segmentation',
    )
    labels.inputs.space = space
    metadata_file = _function(write_metadata, ['metadata'], ['out_file'], 'metadata')
    metadata_file.inputs.metadata = metadata
    tacs = init_pet_tacs_wf(resample_to_segmentation=False)
    workflow.connect(
        [
            (inputnode, registration, [('petref', 'inputnode.petref')]),
            (
                resources,
                registration,
                [('template', 'inputnode.template'), ('mask', 'inputnode.template_mask')],
            ),
            (resources, labels, [('segmentation', 'segmentation'), ('labels', 'label_table')]),
            (resample, labels, [('outputnode.resampling_reference', 'reference')]),
            (resample, tacs, [('outputnode.pet_file', 'inputnode.pet_file')]),
            (support_resample, tacs, [('outputnode.pet_file', 'inputnode.support')]),
            (labels, tacs, [('segmentation', 'inputnode.segmentation')]),
            (resources, tacs, [('labels', 'inputnode.dseg_tsv')]),
            (metadata_file, tacs, [('out_file', 'inputnode.metadata')]),
        ]
    )
    for sampler in () if resampling_cache else (resample, support_resample):
        workflow.connect(
            [
                (
                    inputnode,
                    sampler,
                    [('petref', 'inputnode.pet_ref_file'), ('motion_xfm', 'inputnode.motion_xfm')],
                ),
                (
                    resources,
                    sampler,
                    [('template', 'inputnode.target_ref_file'), ('mask', 'inputnode.target_mask')],
                ),
                (
                    registration,
                    sampler,
                    [('outputnode.petref2template_xfm', 'inputnode.petref2target_xfm')],
                ),
            ]
        )

    entities = {'space': space, 'resolution': resolution}
    atlas_entities = {**entities, 'seg': config.workflow.seg, 'allowed_entities': ('seg',)}
    ds_pet = _sink(
        pet_file,
        'ds_pet',
        suffix='pet',
        desc='preproc',
        extension='.nii.gz',
        compress=True,
        **entities,
    )
    ds_seg = _sink(
        pet_file, 'ds_seg', suffix='dseg', extension='.nii.gz', compress=True, **atlas_entities
    )
    ds_labels = _sink(
        pet_file, 'ds_labels', suffix='dseg', extension='.tsv', check_hdr=False, **atlas_entities
    )
    ds_morph = _sink(
        pet_file, 'ds_morph', suffix='morph', extension='.tsv', check_hdr=False, **atlas_entities
    )
    ds_tacs = _sink(
        pet_file,
        'ds_tacs',
        suffix='tacs',
        desc='preproc',
        extension='.tsv',
        check_hdr=False,
        meta_dict={
            **metadata,
            'ExtractionSpace': space,
            'AtlasSourceSpace': space,
            'CoveragePolicy': 'n/a when any voxel in a region lacks measured '
            'support in that frame',
        },
        **atlas_entities,
    )
    ds_support = _sink(
        pet_file,
        'ds_support',
        suffix='probseg',
        desc='support',
        extension='.nii.gz',
        compress=True,
        **entities,
    )
    ds_forward = _sink(
        pet_file,
        'ds_forward',
        suffix='xfm',
        extension='.h5',
        mode='image',
        **{'from': 'petref', 'to': space},
    )
    ds_inverse = _sink(
        pet_file,
        'ds_inverse',
        suffix='xfm',
        extension='.h5',
        mode='image',
        **{'from': space, 'to': 'petref'},
    )
    # Different target resolutions can yield different estimates; keep transforms distinct.
    ds_forward.inputs.resolution = resolution
    ds_inverse.inputs.resolution = resolution
    ds_registration = _sink(
        pet_file,
        'ds_registration',
        suffix='pet',
        desc='registration',
        extension='.json',
        check_hdr=False,
        **entities,
    )
    ds_warped_ref = _sink(
        pet_file,
        'ds_warped_ref',
        suffix='petref',
        desc=config.workflow.petref,
        extension='.nii.gz',
        compress=True,
        **entities,
    )
    report_metadata = _function(read_metadata, ['in_file'], ['metadata'], 'report_metadata')
    provenance_file = _function(write_metadata, ['metadata'], ['out_file'], 'provenance')
    ds_resources = _sink(
        pet_file,
        'ds_resources',
        suffix='pet',
        desc='resources',
        extension='.json',
        check_hdr=False,
        **entities,
    )
    overlay = pe.Node(SimpleBeforeAfterRPT(before_label='PET', after_label=space), name='overlay')
    ds_overlay = _sink(pet_file, 'ds_overlay', suffix='pet', desc='petnorm', **entities)
    ds_overlay.inputs.datatype = 'figures'
    workflow.connect(
        [
            (labels, ds_seg, [('segmentation', 'in_file')]),
            (resources, ds_labels, [('labels', 'in_file')]),
            (labels, ds_morph, [('morph', 'in_file'), ('morph_metadata', 'meta_dict')]),
            (tacs, ds_tacs, [('outputnode.timeseries', 'in_file')]),
            (resources, provenance_file, [('provenance', 'metadata')]),
            (provenance_file, ds_resources, [('out_file', 'in_file')]),
            (resources, ds_resources, [('provenance', 'meta_dict')]),
            (registration, overlay, [('outputnode.warped_petref', 'before')]),
            (resources, overlay, [('template', 'after')]),
            (overlay, ds_overlay, [('out_report', 'in_file')]),
        ]
    )
    if not registration_cache or not resampling_cache:
        registration_sources = _function(
            registration_metadata,
            [
                'petref',
                'fixed_image',
                'fixed_mask',
                'reference_settings',
                'space',
                'res',
                'sloppy',
            ],
            ['metadata'],
            'registration_sources',
        )
        registration_sources.inputs.reference_settings = reference_metadata(
            metadata, config.workflow.petref, config.workflow.hmc_off
        )
        registration_sources.inputs.space = space
        registration_sources.inputs.res = resolution
        registration_sources.inputs.sloppy = config.execution.sloppy
        workflow.connect(
            [
                (inputnode, registration_sources, [('petref', 'petref')]),
                (
                    resources,
                    registration_sources,
                    [('template', 'fixed_image'), ('mask', 'fixed_mask')],
                ),
            ]
        )
    if not registration_cache:
        workflow.connect(
            [
                (registration, ds_forward, [('outputnode.petref2template_xfm', 'in_file')]),
                (registration, ds_inverse, [('outputnode.template2petref_xfm', 'in_file')]),
                (registration, ds_warped_ref, [('outputnode.warped_petref', 'in_file')]),
                (registration, ds_registration, [('outputnode.report', 'in_file')]),
                (registration, report_metadata, [('outputnode.report', 'in_file')]),
                (report_metadata, ds_registration, [('metadata', 'meta_dict')]),
                (registration_sources, ds_forward, [('metadata', 'meta_dict')]),
                (registration_sources, ds_inverse, [('metadata', 'meta_dict')]),
                (registration_sources, ds_warped_ref, [('metadata', 'meta_dict')]),
            ]
        )
    if not resampling_cache:
        resampling_sources = _function(
            resampling_metadata,
            ['pet_file', 'transform', 'registration', 'metadata'],
            ['metadata'],
            'resampling_sources',
        )
        resampling_sources.inputs.pet_file = pet_file
        resampling_sources.inputs.metadata = metadata
        if registration_cache:
            resampling_sources.inputs.transform = registration_cache['forward']
        else:
            workflow.connect(ds_forward, 'out_file', resampling_sources, 'transform')
        workflow.connect(
            [
                (resample, ds_pet, [('outputnode.pet_file', 'in_file')]),
                (support_resample, ds_support, [('outputnode.pet_file', 'in_file')]),
                (registration_sources, resampling_sources, [('metadata', 'registration')]),
                (resampling_sources, ds_pet, [('metadata', 'meta_dict')]),
                (resampling_sources, ds_support, [('metadata', 'meta_dict')]),
            ]
        )
    return workflow


def init_pet_only_wf(*, pet_file, precomputed=None):
    from .base import _get_wf_name

    workflow = Workflow(name=_get_wf_name(pet_file, 'petonly'))
    metadata = config.execution.layout.get_metadata(pet_file)
    reference_cache = find_reference(
        pet_file,
        metadata,
        precomputed,
        config.execution.derivatives,
        config.workflow.petref,
        config.workflow.hmc_off,
    )
    prepare = init_pet_only_prepare_wf(
        pet_file=pet_file,
        metadata=metadata,
        precomputed=precomputed,
        reference_cache=reference_cache,
    )
    for index, reference in enumerate(config.workflow.spaces.references):
        registration_cache, resampling_cache = find_target_derivatives(
            pet_file,
            metadata,
            reference_cache,
            config.execution.derivatives,
            reference.space,
            reference.spec,
            config.workflow.petref,
            config.workflow.hmc_off,
            config.execution.sloppy,
        )
        target = init_pet_only_target_wf(
            pet_file=pet_file,
            metadata=metadata,
            reference=reference,
            name=f'target_{index}_wf',
            registration_cache=registration_cache,
            resampling_cache=resampling_cache,
        )
        workflow.connect(
            [
                (
                    prepare,
                    target,
                    [
                        ('outputnode.pet_file', 'inputnode.pet_file'),
                        ('outputnode.petref', 'inputnode.petref'),
                        ('outputnode.motion_xfm', 'inputnode.motion_xfm'),
                    ],
                )
            ]
        )
    return workflow


def init_pet_only_subject_wf(*, subject_id, session_id=None):
    from ...utils.bids import collect_subject_data
    from ..base import _session_bids_filters, _stringify_sessions

    data = collect_subject_data(
        config.execution.bids_dir,
        subject_id,
        session_id=session_id,
        bids_filters=_session_bids_filters(session_id),
    )
    runs = data['pet']
    if not runs:
        raise ValueError(f'No PET data found for subject {subject_id}.')
    files = []
    for run in runs:
        series = [run] if isinstance(run, str) else run
        if len(series) != 1:
            raise ValueError('PET-only processing currently requires one image per acquisition.')
        files.extend(series)
    ses = _stringify_sessions(session_id)
    workflow = Workflow(name=f'sub_{subject_id}' + (f'_ses_{ses}' if ses else '') + '_wf')
    workflow.__desc__ = (
        f'PET-only processing was performed with PETPrep {config.environment.version}. '
        'Subject anatomical processing was omitted. Regional TACs were extracted '
        'from normalized PET using template-space labels. Reported atlas volumes '
        'describe template regions, not individual anatomy.'
    )
    for pet_file in files:
        pet_cache = collect_pet_only_derivatives(pet_file, config.execution.derivatives)
        workflow.add_nodes([init_pet_only_wf(pet_file=pet_file, precomputed=pet_cache)])
    summary = pe.Node(
        SubjectSummary(
            subject_id=subject_id,
            pet=files,
            t1w=[],
            t2w=[],
            std_spaces=config.workflow.spaces.get_spaces(),
            nstd_spaces=[],
        ),
        name='summary',
    )
    about = pe.Node(
        AboutSummary(version=config.environment.version, command=' '.join(sys.argv)), name='about'
    )
    for node, desc in ((summary, 'summary'), (about, 'about')):
        sink = _sink(files[0], f'ds_{desc}', suffix='pet', desc=desc)
        sink.inputs.datatype = 'figures'
        workflow.connect(node, 'out_report', sink, 'in_file')
    return workflow
