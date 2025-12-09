# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
from collections.abc import Sequence
from pathlib import Path

import nibabel as nb
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nitransforms.linear import Affine, LinearTransformsMapping
from niworkflows.interfaces.header import ValidateImage
from niworkflows.utils.connections import listify

from ... import config
from ...data import load as load_data
from ...interfaces import DerivativesDataSink
from ...interfaces.reports import FunctionalSummary
from ...interfaces.resampling import ResampleSeries
from ...utils.misc import estimate_pet_mem_usage

# PET workflows
from .hmc import init_pet_hmc_wf
from .outputs import (
    init_ds_hmc_wf,
    init_ds_petmask_wf,
    init_ds_petref_wf,
    init_ds_refmask_wf,
    init_ds_registration_wf,
    init_func_fit_reports_wf,
    init_refmask_report_wf,
    prepare_timing_parameters,
)
from .ref_tacs import init_pet_ref_tacs_wf
from .reference_mask import init_pet_refmask_wf
from .registration import init_pet_reg_wf


def _extract_twa_image(
    pet_file: str,
    output_dir: 'Path',
    frame_start_times: 'Sequence[float] | None',
    frame_durations: 'Sequence[float] | None',
) -> str:
    """Return a time-weighted average (twa) reference image from a 4D PET series."""

    from pathlib import Path

    import nibabel as nb
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img = nb.load(pet_file)
    if img.ndim < 4 or img.shape[-1] == 1:
        return pet_file

    if frame_start_times is None or frame_durations is None:
        raise ValueError(
            'Frame timing metadata are required to compute a time-weighted reference image.'
        )

    frame_start_times = np.asarray(frame_start_times, dtype=float)
    frame_durations = np.asarray(frame_durations, dtype=float)

    if frame_start_times.ndim != 1 or frame_durations.ndim != 1:
        raise ValueError('Frame timing metadata must be one-dimensional sequences.')

    if len(frame_start_times) != len(frame_durations):
        raise ValueError('FrameTimesStart and FrameDuration must have the same length.')

    if len(frame_durations) != img.shape[-1]:
        raise ValueError(
            'Frame timing metadata must match the number of frames in the PET series.'
        )

    if np.any(frame_durations <= 0):
        raise ValueError('FrameDuration values must all be positive.')

    if np.any(np.diff(frame_start_times) < 0):
        raise ValueError('FrameTimesStart values must be non-decreasing.')

    hdr = img.header.copy()
    data = np.asanyarray(img.dataobj)
    weighted_average = np.average(data, axis=-1, weights=frame_durations).astype(np.float32)
    hdr.set_data_shape(weighted_average.shape)

    pet_path = Path(pet_file)
    # Drop all suffixes (e.g., `.nii.gz`) before appending the reference label
    pet_stem = pet_path
    while pet_stem.suffix:
        pet_stem = pet_stem.with_suffix('')

    out_file = output_dir / f'{pet_stem.name}_timeavgref.nii.gz'
    img.__class__(weighted_average, img.affine, hdr).to_filename(out_file)
    return str(out_file)


def _extract_first5min_image(
    pet_file: str,
    output_dir: 'Path',
    frame_start_times: 'Sequence[float] | None',
    frame_durations: 'Sequence[float] | None',
    window_sec: float = 300.0,
) -> str:
    """Average the early (0-``window_sec``) portion of a PET series."""

    from pathlib import Path

    import nibabel as nb
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img = nb.load(pet_file)
    if img.ndim < 4 or img.shape[-1] == 1:
        return pet_file

    if frame_start_times is None or frame_durations is None:
        raise ValueError('Frame timing metadata are required to compute an early reference image.')

    frame_start_times = np.asarray(frame_start_times, dtype=float)
    frame_durations = np.asarray(frame_durations, dtype=float)

    if frame_start_times.ndim != 1 or frame_durations.ndim != 1:
        raise ValueError('Frame timing metadata must be one-dimensional sequences.')

    if len(frame_start_times) != len(frame_durations):
        raise ValueError('FrameTimesStart and FrameDuration must have the same length.')

    if len(frame_durations) != img.shape[-1]:
        raise ValueError(
            'Frame timing metadata must match the number of frames in the PET series.'
        )

    if np.any(frame_durations <= 0):
        raise ValueError('FrameDuration values must all be positive.')

    if np.any(np.diff(frame_start_times) < 0):
        raise ValueError('FrameTimesStart values must be non-decreasing.')

    frame_ends = frame_start_times + frame_durations
    included_durations = np.clip(frame_ends, 0.0, window_sec) - np.clip(
        frame_start_times, 0.0, window_sec
    )

    if not np.any(included_durations > 0):
        raise ValueError('No frames overlap with the first 5 minutes of the acquisition.')

    hdr = img.header.copy()
    data = np.asanyarray(img.dataobj)
    weighted_average = np.average(data, axis=-1, weights=included_durations).astype(np.float32)
    hdr.set_data_shape(weighted_average.shape)

    pet_path = Path(pet_file)
    while pet_path.suffix:
        pet_path = pet_path.with_suffix('')

    out_file = output_dir / f'{pet_path.name}_first5minref.nii.gz'
    img.__class__(weighted_average, img.affine, hdr).to_filename(out_file)
    return str(out_file)


def _extract_sum_image(pet_file: str, output_dir: 'Path') -> str:
    """Return a summed reference image from a 4D PET series."""

    from pathlib import Path

    import nibabel as nb
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    img = nb.load(pet_file)
    if img.ndim < 4 or img.shape[-1] == 1:
        return pet_file

    hdr = img.header.copy()
    data = np.asanyarray(img.dataobj)
    summed = data.sum(axis=-1, dtype=np.float64).astype(np.float32)
    hdr.set_data_shape(summed.shape)

    pet_path = Path(pet_file)
    pet_stem = pet_path
    while pet_stem.suffix:
        pet_stem = pet_stem.with_suffix('')

    out_file = output_dir / f'{pet_stem.name}_sumref.nii.gz'
    img.__class__(summed, img.affine, hdr).to_filename(out_file)
    return str(out_file)


def _select_best_petref(labels, scores, transforms, inv_transforms, winners, petrefs):
    """Select the PET reference with the lowest registration cost."""

    if not labels or not scores:
        raise ValueError('No PET reference candidates were provided for selection.')

    best_idx = None
    best_score = float('inf')
    for idx, score in enumerate(scores):
        if score is None:
            continue
        if score < best_score:
            best_idx = idx
            best_score = score

    if best_idx is None:
        raise ValueError('No registration scores were available for selection.')

    return (
        labels[best_idx],
        best_score,
        transforms[best_idx],
        inv_transforms[best_idx],
        winners[best_idx],
        petrefs[best_idx],
    )


def _write_identity_xforms(num_frames: int, filename: Path) -> Path:
    """Write ``num_frames`` identity transforms to ``filename``."""

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    n_xforms = max(int(num_frames or 0), 1)
    LinearTransformsMapping([Affine() for _ in range(n_xforms)]).to_filename(filename, fmt='itk')
    return filename


def init_pet_fit_wf(
    *,
    pet_series: list[str],
    precomputed: dict = None,
    omp_nthreads: int = 1,
    name: str = 'pet_fit_wf',
) -> pe.Workflow:
    """
    This workflow controls the minimal estimation steps for functional preprocessing.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.tests import mock_config
            from petprep import config
            from petprep.workflows.pet.fit import init_pet_fit_wf
            with mock_config():
                pet_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_pet.nii.gz"
                wf = init_pet_fit_wf(pet_series=[str(pet_file)])

    Parameters
    ----------
    pet_series
        List of paths to NIfTI files
    precomputed
        Dictionary containing precomputed derivatives to reuse, if possible.

    Inputs
    ------
    pet_file
        PET series NIfTI file
    t1w_preproc
        Bias-corrected structural template image
    t1w_mask
        Mask of the skull-stripped template image
    t1w_dseg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    anat2std_xfm
        List of transform files, collated with templates
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
    segmentation
        Segmentation file in T1w space
    dseg_tsv
        TSV with segmentation statistics

    Outputs
    -------
    petref
        PET reference image used for head motion correction.
    pet_mask
        Mask of ``petref``.
    motion_xfm
        Affine transforms from each PET volume to ``petref``, written
        as concatenated ITK affine transforms.
    petref2anat_xfm
        Affine transform mapping from PET reference space to the anatomical
        space.

    See Also
    --------

    * :py:func:`~petprep.workflows.pet.hmc.init_pet_hmc_wf`
    * :py:func:`~petprep.workflows.pet.registration.init_pet_reg_wf`
    * :py:func:`~petprep.workflows.pet.outputs.init_ds_petref_wf`
    * :py:func:`~petprep.workflows.pet.outputs.init_ds_hmc_wf`
    * :py:func:`~petprep.workflows.pet.outputs.init_ds_registration_wf`

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from petprep.utils.misc import estimate_pet_mem_usage

    if precomputed is None:
        precomputed = {}
    pet_series = listify(pet_series)
    layout = config.execution.layout

    pet_file = pet_series[0]

    # Get metadata from PET file(s)
    metadata = layout.get_metadata(pet_file)
    orientation = ''.join(nb.aff2axcodes(nb.load(pet_file).affine))

    pet_tlen, mem_gb = estimate_pet_mem_usage(pet_file)

    petref = precomputed.get('petref')
    # Can contain
    #  1) petref2anat
    #  2) hmc
    transforms = precomputed.get('transforms', {})
    hmc_xforms = transforms.get('hmc')
    petref2anat_xform = transforms.get('petref2anat')

    if (petref is None) ^ (hmc_xforms is None):
        raise ValueError("Both 'petref' and 'hmc' transforms must be provided together.")

    if config.workflow.hmc_off and (petref or hmc_xforms):
        config.loggers.workflow.warning(
            'Ignoring precomputed motion correction derivatives because --hmc-off was set.'
        )
        petref = None
        hmc_xforms = None

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_file',
                # Anatomical coregistration
                't1w_preproc',
                't1w_mask',
                't1w_dseg',
                't1w_tpms',
                'subjects_dir',
                'subject_id',
                'fsnative2t1w_xfm',
                'segmentation',
                'dseg_tsv',
            ],
        ),
        name='inputnode',
    )
    inputnode.inputs.pet_file = pet_series

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'petref',
                'pet_mask',
                'motion_xfm',
                'petref2anat_xfm',
                'refmask',
            ],
        ),
        name='outputnode',
    )

    # If all derivatives exist, inputnode could go unconnected, so add explicitly
    workflow.add_nodes([inputnode])

    petref_buffer = pe.Node(
        niu.IdentityInterface(fields=['petref', 'pet_file']),
        name='petref_buffer',
    )
    hmc_buffer = pe.Node(niu.IdentityInterface(fields=['hmc_xforms']), name='hmc_buffer')

    timing_parameters = prepare_timing_parameters(metadata)
    frame_durations = timing_parameters.get('FrameDuration')
    frame_start_times = timing_parameters.get('FrameTimesStart')

    if frame_durations is None or frame_start_times is None:
        raise ValueError(
            'Metadata is missing required frame timing information: '
            "'FrameDuration' or 'FrameTimesStart'. "
            'Please check your BIDS JSON sidecar.'
        )

    requested_petref_strategy = getattr(config.workflow, 'petref', 'template')
    hmc_disabled = bool(config.workflow.hmc_off)
    petref_strategy = requested_petref_strategy
    petref_candidates = None
    petref_candidate_labels: list[str] = []
    if requested_petref_strategy == 'auto':
        petref_strategy = 'auto'
        petref_candidate_labels = ['template', 'twa', 'sum', 'first5min']
        petref_candidates = pe.Node(
            niu.IdentityInterface(fields=petref_candidate_labels), name='petref_candidates'
        )

    if hmc_disabled and petref_strategy == 'template':
        config.loggers.workflow.warning(
            'Head motion correction disabled (--hmc-off); using a time-weighted average '
            'reference instead of the motion correction template.'
        )
        petref_strategy = 'twa'

    use_corrected_reference = petref_strategy in {'twa', 'sum', 'first5min', 'auto'}
    reference_function = _extract_twa_image
    reference_kwargs: dict[str, object] = {
        'output_dir': config.execution.work_dir,
        'frame_start_times': frame_start_times,
        'frame_durations': frame_durations,
    }
    reference_node_name = 'twa_reference'
    reference_input_names = ['pet_file', 'output_dir', 'frame_start_times', 'frame_durations']
    report_reference_input_names = reference_input_names

    if petref_strategy == 'sum':
        reference_function = _extract_sum_image
        reference_kwargs = {'output_dir': config.execution.work_dir}
        reference_node_name = 'sum_reference'
        reference_input_names = ['pet_file', 'output_dir']

    if petref_strategy == 'first5min':
        reference_function = _extract_first5min_image
        reference_node_name = 'first5min_reference'

    corrected_pet_for_report = None
    corrected_reference = None

    requires_report_reference = pet_tlen > 1
    report_pet_for_coreg = None
    report_pet_reference = None

    if requires_report_reference:
        report_pet_for_coreg = pe.Node(
            ResampleSeries(),
            name='report_pet_for_coreg',
            n_procs=omp_nthreads,
            mem_gb=mem_gb['resampled'],
        )
        report_pet_reference = pe.Node(
            niu.Function(
                function=_extract_twa_image,
                input_names=report_reference_input_names,
                output_names=['out_file'],
            ),
            name='report_petref',
        )
        report_pet_reference.inputs.output_dir = config.execution.work_dir
        report_pet_reference.inputs.frame_start_times = frame_start_times
        report_pet_reference.inputs.frame_durations = frame_durations

    if use_corrected_reference:
        corrected_pet_for_report = pe.Node(
            ResampleSeries(),
            name='corrected_pet_for_report',
            n_procs=omp_nthreads,
            mem_gb=mem_gb['resampled'],
        )
        corrected_reference = pe.Node(
            niu.Function(
                function=reference_function,
                input_names=reference_input_names,
                output_names=['out_file'],
            ),
            name=reference_node_name,
        )
        corrected_reference.inputs.output_dir = config.execution.work_dir
        if petref_strategy in {'twa', 'first5min'}:
            corrected_reference.inputs.frame_start_times = frame_start_times
            corrected_reference.inputs.frame_durations = frame_durations

    reference_nodes: dict[str, pe.Node] = {}
    if petref_strategy == 'auto':
        reference_nodes['twa'] = pe.Node(
            niu.Function(
                function=_extract_twa_image,
                input_names=reference_input_names,
                output_names=['out_file'],
            ),
            name='auto_twa_reference',
        )
        reference_nodes['twa'].inputs.output_dir = config.execution.work_dir
        reference_nodes['twa'].inputs.frame_start_times = frame_start_times
        reference_nodes['twa'].inputs.frame_durations = frame_durations

        reference_nodes['sum'] = pe.Node(
            niu.Function(
                function=_extract_sum_image,
                input_names=['pet_file', 'output_dir'],
                output_names=['out_file'],
            ),
            name='auto_sum_reference',
        )
        reference_nodes['sum'].inputs.output_dir = config.execution.work_dir

        reference_nodes['first5min'] = pe.Node(
            niu.Function(
                function=_extract_first5min_image,
                input_names=reference_input_names,
                output_names=['out_file'],
            ),
            name='auto_first5min_reference',
        )
        reference_nodes['first5min'].inputs.output_dir = config.execution.work_dir
        reference_nodes['first5min'].inputs.frame_start_times = frame_start_times
        reference_nodes['first5min'].inputs.frame_durations = frame_durations

    registration_method = 'Precomputed'
    if not petref2anat_xform:
        registration_method = {
            'mri_coreg': 'mri_coreg',
            'robust': 'mri_robust_register',
            'ants': 'ants_registration',
            'auto': 'auto_select',
        }[config.workflow.pet2anat_method]
    if hmc_disabled:
        config.execution.work_dir.mkdir(parents=True, exist_ok=True)
        petref = petref or reference_function(pet_file, **reference_kwargs)
        idmat_fname = config.execution.work_dir / 'idmat.tfm'
        n_frames = len(frame_durations)
        hmc_xforms = _write_identity_xforms(n_frames, idmat_fname)
        config.loggers.workflow.info('Head motion correction disabled; using identity transforms.')
        if petref_strategy == 'auto' and petref_candidates is not None:
            petref_candidates.inputs.template = petref

    if pet_tlen <= 1:  # 3D PET
        petref = pet_file
        idmat_fname = config.execution.work_dir / 'idmat.tfm'
        hmc_xforms = _write_identity_xforms(pet_tlen, idmat_fname)
        config.loggers.workflow.debug('3D PET file - motion correction not needed')
    if petref:
        petref_buffer.inputs.petref = petref
        config.loggers.workflow.debug(f'(Re)using motion correction reference: {petref}')
    if hmc_xforms:
        hmc_buffer.inputs.hmc_xforms = hmc_xforms
        config.loggers.workflow.debug(f'(Re)using motion correction transforms: {hmc_xforms}')

    summary = pe.Node(
        FunctionalSummary(
            registration=registration_method,
            registration_dof=config.workflow.pet2anat_dof,
            orientation=orientation,
            metadata=metadata,
            petref_strategy=petref_strategy,
            requested_petref_strategy=requested_petref_strategy,
            hmc_disabled=hmc_disabled,
        ),
        name='summary',
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    func_fit_reports_wf = init_func_fit_reports_wf(
        freesurfer=config.workflow.run_reconall,
        output_dir=config.execution.petprep_dir,
        ref_name=config.workflow.ref_mask_name,
    )

    workflow.connect([
        (petref_buffer, outputnode, [
            ('petref', 'petref'),
        ]),
        (hmc_buffer, outputnode, [
            ('hmc_xforms', 'motion_xfm'),
        ]),
        (inputnode, func_fit_reports_wf, [
            ('pet_file', 'inputnode.source_file'),
            ('t1w_preproc', 'inputnode.t1w_preproc'),
            # May not need all of these
            ('t1w_mask', 'inputnode.t1w_mask'),
            ('t1w_dseg', 'inputnode.t1w_dseg'),
            ('subjects_dir', 'inputnode.subjects_dir'),
            ('subject_id', 'inputnode.subject_id'),
        ]),
        (petref_buffer, func_fit_reports_wf, [('petref', 'inputnode.petref')]),
        (outputnode, func_fit_reports_wf, [
            ('pet_mask', 'inputnode.pet_mask'),
            ('petref2anat_xfm', 'inputnode.petref2anat_xfm'),
        ]),
        (summary, func_fit_reports_wf, [('out_report', 'inputnode.summary_report')]),
    ])  # fmt:skip

    # Stage 1: Estimate head motion and reference image
    if not hmc_xforms:
        config.loggers.workflow.info(
            'PET Stage 1: Adding motion correction workflow and petref estimation'
        )
        pet_hmc_wf = init_pet_hmc_wf(
            name='pet_hmc_wf',
            mem_gb=mem_gb['filesize'],
            omp_nthreads=omp_nthreads,
            fwhm=config.workflow.hmc_fwhm,
            start_time=config.workflow.hmc_start_time,
            frame_durations=frame_durations,
            frame_start_times=frame_start_times,
            initial_frame=config.workflow.hmc_init_frame,
            fixed_frame=config.workflow.hmc_fix_frame,
        )

        ds_hmc_wf = init_ds_hmc_wf(
            bids_root=layout.root,
            output_dir=config.execution.petprep_dir,
        )
        ds_hmc_wf.inputs.inputnode.source_files = [pet_file]

        ds_petref_wf = init_ds_petref_wf(
            bids_root=layout.root,
            output_dir=config.execution.petprep_dir,
            desc='hmc',
            name='ds_petref_wf',
        )
        ds_petref_wf.inputs.inputnode.source_files = [pet_file]

        # Validation node for the original PET file
        val_pet = pe.Node(ValidateImage(), name='val_pet')
        val_pet.inputs.in_file = pet_file

        workflow.connect([
            (val_pet, petref_buffer, [('out_file', 'pet_file')]),
            (val_pet, func_fit_reports_wf, [('out_report', 'inputnode.validation_report')]),
            (val_pet, pet_hmc_wf, [
                ('out_file', 'inputnode.pet_file'),
            ]),
            (pet_hmc_wf, ds_hmc_wf, [('outputnode.xforms', 'inputnode.xforms')]),
            (ds_hmc_wf, hmc_buffer, [('outputnode.xforms', 'hmc_xforms')]),
            (pet_hmc_wf, ds_petref_wf, [('outputnode.petref', 'inputnode.petref')]),
        ])  # fmt:skip

        if petref_strategy == 'auto':
            workflow.connect([(pet_hmc_wf, petref_candidates, [('outputnode.petref', 'template')])])
        elif use_corrected_reference:
            workflow.connect([
                (pet_hmc_wf, corrected_pet_for_report, [('outputnode.petref', 'ref_file')]),
                (val_pet, corrected_pet_for_report, [('out_file', 'in_file')]),
                (hmc_buffer, corrected_pet_for_report, [('hmc_xforms', 'transforms')]),
                (corrected_pet_for_report, corrected_reference, [('out_file', 'pet_file')]),
                (corrected_reference, petref_buffer, [('out_file', 'petref')]),
            ])  # fmt:skip
        else:
            workflow.connect([(pet_hmc_wf, petref_buffer, [('outputnode.petref', 'petref')])])

        if petref_strategy == 'auto' and use_corrected_reference:
            workflow.connect([
                (pet_hmc_wf, corrected_pet_for_report, [('outputnode.petref', 'ref_file')]),
                (val_pet, corrected_pet_for_report, [('out_file', 'in_file')]),
                (hmc_buffer, corrected_pet_for_report, [('hmc_xforms', 'transforms')]),
                (corrected_pet_for_report, reference_nodes['twa'], [('out_file', 'pet_file')]),
                (corrected_pet_for_report, reference_nodes['sum'], [('out_file', 'pet_file')]),
                (corrected_pet_for_report, reference_nodes['first5min'], [('out_file', 'pet_file')]),
                (reference_nodes['twa'], petref_candidates, [('out_file', 'twa')]),
                (reference_nodes['sum'], petref_candidates, [('out_file', 'sum')]),
                (reference_nodes['first5min'], petref_candidates, [('out_file', 'first5min')]),
            ])  # fmt:skip

        if report_pet_reference:
            workflow.connect([
                (pet_hmc_wf, report_pet_for_coreg, [('outputnode.petref', 'ref_file')]),
                (val_pet, report_pet_for_coreg, [('out_file', 'in_file')]),
                (hmc_buffer, report_pet_for_coreg, [('hmc_xforms', 'transforms')]),
                (report_pet_for_coreg, report_pet_reference, [('out_file', 'pet_file')]),
            ])  # fmt:skip
    else:
        config.loggers.workflow.info(
            'PET Stage 1: Found head motion correction transforms and petref - skipping Stage 1'
        )

        val_pet = pe.Node(ValidateImage(), name='val_pet')

        workflow.connect([
            (val_pet, petref_buffer, [('out_file', 'pet_file')]),
            (val_pet, func_fit_reports_wf, [('out_report', 'inputnode.validation_report')]),

        ])  # fmt:skip
        val_pet.inputs.in_file = pet_file
        if petref_strategy == 'auto':
            corrected_pet_for_report.inputs.ref_file = petref
            petref_candidates.inputs.template = petref
            workflow.connect([
                (val_pet, corrected_pet_for_report, [('out_file', 'in_file')]),
                (hmc_buffer, corrected_pet_for_report, [('hmc_xforms', 'transforms')]),
                (corrected_pet_for_report, reference_nodes['twa'], [('out_file', 'pet_file')]),
                (corrected_pet_for_report, reference_nodes['sum'], [('out_file', 'pet_file')]),
                (corrected_pet_for_report, reference_nodes['first5min'], [('out_file', 'pet_file')]),
                (reference_nodes['twa'], petref_candidates, [('out_file', 'twa')]),
                (reference_nodes['sum'], petref_candidates, [('out_file', 'sum')]),
                (reference_nodes['first5min'], petref_candidates, [('out_file', 'first5min')]),
            ])  # fmt:skip
        elif use_corrected_reference:
            corrected_pet_for_report.inputs.ref_file = petref
            workflow.connect(
                [
                    (val_pet, corrected_pet_for_report, [('out_file', 'in_file')]),
                    (hmc_buffer, corrected_pet_for_report, [('hmc_xforms', 'transforms')]),
                    (corrected_pet_for_report, corrected_reference, [('out_file', 'pet_file')]),
                    (corrected_reference, petref_buffer, [('out_file', 'petref')]),
                ]
            )  # fmt:skip
        else:
            petref_buffer.inputs.petref = petref

        if report_pet_reference:
            workflow.connect([
                (petref_buffer, report_pet_for_coreg, [('petref', 'ref_file')]),
                (val_pet, report_pet_for_coreg, [('out_file', 'in_file')]),
                (hmc_buffer, report_pet_for_coreg, [('hmc_xforms', 'transforms')]),
                (report_pet_for_coreg, report_pet_reference, [('out_file', 'pet_file')]),
            ])  # fmt:skip

    if report_pet_reference:
        workflow.connect([
            (report_pet_reference, func_fit_reports_wf, [('out_file', 'inputnode.report_pet')])
        ])  # fmt:skip
    else:
        func_fit_reports_wf.inputs.inputnode.report_pet = pet_file

    # Stage 2: Coregistration
    rerun_coreg = petref2anat_xform and (
        config.workflow.petref_specified or config.workflow.pet2anat_method_specified
    )
    if rerun_coreg:
        config.loggers.workflow.info(
            'PET Stage 2: Re-running co-registration because --petref or --pet2anat-method '
            'were explicitly requested.'
        )
        petref2anat_xform = None

    pet_to_t1_source = None
    pet_to_t1_field = None

    if not petref2anat_xform:
        config.loggers.workflow.info('PET Stage 2: Adding co-registration workflow of PET to T1w')

        if petref_strategy == 'auto':
            ds_petreg_wf = init_ds_registration_wf(
                bids_root=layout.root,
                output_dir=config.execution.petprep_dir,
                source='petref',
                dest='T1w',
                name='ds_petreg_wf',
            )

            score_merge = pe.Node(niu.Merge(len(petref_candidate_labels)), name='merge_scores')
            xfm_merge = pe.Node(niu.Merge(len(petref_candidate_labels)), name='merge_xfms')
            inv_merge = pe.Node(niu.Merge(len(petref_candidate_labels)), name='merge_inv_xfms')
            winner_merge = pe.Node(
                niu.Merge(len(petref_candidate_labels)), name='merge_reg_winners'
            )
            label_merge = pe.Node(niu.Merge(len(petref_candidate_labels)), name='merge_labels')
            petref_merge = pe.Node(niu.Merge(len(petref_candidate_labels)), name='merge_petrefs')

            for idx, label in enumerate(petref_candidate_labels):
                reg_wf = init_pet_reg_wf(
                    pet2anat_dof=config.workflow.pet2anat_dof,
                    omp_nthreads=omp_nthreads,
                    mem_gb=mem_gb['resampled'],
                    pet2anat_method=config.workflow.pet2anat_method,
                    sloppy=config.execution.sloppy,
                    name=f'pet_reg_wf_{label}',
                )

                label_src = pe.Node(niu.IdentityInterface(fields=['label']), name=f'label_{label}')
                label_src.inputs.label = label

                workflow.connect([
                    (inputnode, reg_wf, [
                        ('t1w_preproc', 'inputnode.anat_preproc'),
                        ('t1w_mask', 'inputnode.anat_mask'),
                    ]),
                    (petref_candidates, reg_wf, [(label, 'inputnode.ref_pet_brain')]),
                    (reg_wf, score_merge, [(
                        'outputnode.registration_score', f'in{idx + 1}'
                    )]),
                    (reg_wf, xfm_merge, [('outputnode.itk_pet_to_t1', f'in{idx + 1}')]),
                    (reg_wf, inv_merge, [('outputnode.itk_t1_to_pet', f'in{idx + 1}')]),
                    (reg_wf, winner_merge, [('outputnode.registration_winner', f'in{idx + 1}')]),
                    (petref_candidates, petref_merge, [(label, f'in{idx + 1}')]),
                    (label_src, label_merge, [('label', f'in{idx + 1}')]),
                ])  # fmt:skip

            select_best_ref = pe.Node(
                niu.Function(
                    function=_select_best_petref,
                    input_names=[
                        'labels',
                        'scores',
                        'transforms',
                        'inv_transforms',
                        'winners',
                        'petrefs',
                    ],
                    output_names=[
                        'best_label',
                        'best_score',
                        'best_transform',
                        'best_inv_transform',
                        'best_winner',
                        'best_petref',
                    ],
                ),
                name='select_best_petref',
            )

            workflow.connect([
                (score_merge, select_best_ref, [('out', 'scores')]),
                (xfm_merge, select_best_ref, [('out', 'transforms')]),
                (inv_merge, select_best_ref, [('out', 'inv_transforms')]),
                (winner_merge, select_best_ref, [('out', 'winners')]),
                (label_merge, select_best_ref, [('out', 'labels')]),
                (petref_merge, select_best_ref, [('out', 'petrefs')]),
                (val_pet, ds_petreg_wf, [('out_file', 'inputnode.source_files')]),
                (select_best_ref, ds_petreg_wf, [('best_transform', 'inputnode.xform')]),
                (ds_petreg_wf, outputnode, [('outputnode.xform', 'petref2anat_xfm')]),
                (select_best_ref, petref_buffer, [('best_petref', 'petref')]),
                (select_best_ref, summary, [('best_winner', 'registration_winner')]),
                (select_best_ref, summary, [('best_label', 'petref_strategy')]),
            ])  # fmt:skip

            pet_to_t1_source = select_best_ref
            pet_to_t1_field = 'best_transform'
        else:
            # calculate PET registration to T1w
            pet_reg_wf = init_pet_reg_wf(
                pet2anat_dof=config.workflow.pet2anat_dof,
                omp_nthreads=omp_nthreads,
                mem_gb=mem_gb['resampled'],
                pet2anat_method=config.workflow.pet2anat_method,
                sloppy=config.execution.sloppy,
            )

            ds_petreg_wf = init_ds_registration_wf(
                bids_root=layout.root,
                output_dir=config.execution.petprep_dir,
                source='petref',
                dest='T1w',
                name='ds_petreg_wf',
            )

            workflow.connect([
                (inputnode, pet_reg_wf, [
                    ('t1w_preproc', 'inputnode.anat_preproc'),
                    ('t1w_mask', 'inputnode.anat_mask'),
                ]),
                (petref_buffer, pet_reg_wf, [('petref', 'inputnode.ref_pet_brain')]),
                (val_pet, ds_petreg_wf, [('out_file', 'inputnode.source_files')]),
                (pet_reg_wf, ds_petreg_wf, [('outputnode.itk_pet_to_t1', 'inputnode.xform')]),
                (ds_petreg_wf, outputnode, [('outputnode.xform', 'petref2anat_xfm')]),
                (pet_reg_wf, summary, [('outputnode.registration_winner', 'registration_winner')]),
            ])  # fmt:skip

            pet_to_t1_source = pet_reg_wf
            pet_to_t1_field = 'outputnode.itk_pet_to_t1'
    else:
        outputnode.inputs.petref2anat_xfm = petref2anat_xform

    # Stage 3: Estimate PET brain mask
    config.loggers.workflow.info('PET Stage 3: Adding estimation of PET brain mask')
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

    from .confounds import _binary_union, _smooth_binarize

    t1w_mask_tfm = pe.Node(
        ApplyTransforms(interpolation='MultiLabel', invert_transform_flags=[True]),
        name='t1w_mask_tfm',
    )
    petref_mask = pe.Node(niu.Function(function=_smooth_binarize), name='petref_mask')
    petref_mask.inputs.fwhm = 10.0
    petref_mask.inputs.thresh = 0.2
    merge_mask = pe.Node(niu.Function(function=_binary_union), name='merge_mask')

    if petref2anat_xform:
        t1w_mask_tfm.inputs.transforms = petref2anat_xform
    elif pet_to_t1_source and pet_to_t1_field:
        workflow.connect(
            [(pet_to_t1_source, t1w_mask_tfm, [(pet_to_t1_field, 'transforms')])]
        )

    workflow.connect(
        [
            (inputnode, t1w_mask_tfm, [('t1w_mask', 'input_image')]),
            (petref_buffer, t1w_mask_tfm, [('petref', 'reference_image')]),
            (petref_buffer, petref_mask, [('petref', 'in_file')]),
            (petref_mask, merge_mask, [('out', 'mask1')]),
            (t1w_mask_tfm, merge_mask, [('output_image', 'mask2')]),
            (merge_mask, outputnode, [('out', 'pet_mask')]),
        ]
    )

    ds_petmask_wf = init_ds_petmask_wf(
        output_dir=config.execution.petprep_dir,
        desc='brain',
        name='ds_petmask_wf',
    )
    ds_petmask_wf.inputs.inputnode.source_files = [pet_file]
    workflow.connect([(merge_mask, ds_petmask_wf, [('out', 'inputnode.petmask')])])

    pvc_method = getattr(config.workflow, 'pvc_method', None)

    # Stage 4: Reference mask generation
    if config.workflow.ref_mask_name:
        config.loggers.workflow.info(
            f'PET Stage 4: Generating {config.workflow.ref_mask_name} reference mask'
        )

        refmask_wf = init_pet_refmask_wf(
            segmentation=config.workflow.seg,
            ref_mask_name=config.workflow.ref_mask_name,
            ref_mask_index=list(config.workflow.ref_mask_index)
            if config.workflow.ref_mask_index
            else None,
            config_path=str(load_data('reference_mask/config.json')),
            name='pet_refmask_wf',
        )

        ds_refmask_wf = init_ds_refmask_wf(
            output_dir=config.execution.petprep_dir,
            ref_name=config.workflow.ref_mask_name,
            name='ds_refmask_wf',
        )

        refmask_report_wf = init_refmask_report_wf(
            output_dir=config.execution.petprep_dir,
            ref_name=config.workflow.ref_mask_name,
            name='refmask_report_wf',
        )

        gm_select = pe.Node(niu.Select(index=0), name='select_gm_probseg')

        pet_ref_tacs_wf = init_pet_ref_tacs_wf(name='pet_ref_tacs_wf')
        pet_ref_tacs_wf.inputs.inputnode.metadata = str(
            Path(pet_file).with_suffix('').with_suffix('.json')
        )
        pet_ref_tacs_wf.inputs.inputnode.ref_mask_name = config.workflow.ref_mask_name

        if pvc_method is None:
            ds_ref_tacs = pe.Node(
                DerivativesDataSink(
                    base_directory=config.execution.petprep_dir,
                    suffix='tacs',
                    desc='preproc',
                    label=config.workflow.ref_mask_name,
                    allowed_entities=('label',),
                    TaskName=metadata.get('TaskName'),
                    **timing_parameters,
                ),
                name='ds_ref_tacs',
                run_without_submitting=True,
                mem_gb=config.DEFAULT_MEMORY_MIN_GB,
            )
            ds_ref_tacs.inputs.source_file = pet_file

        workflow.connect([(inputnode, gm_select, [('t1w_tpms', 'inlist')])])

        workflow.connect(
            [
                (
                    inputnode,
                    refmask_wf,
                    [
                        ('segmentation', 'inputnode.seg_file'),
                    ],
                ),
                (
                    gm_select,
                    refmask_wf,
                    [('out', 'inputnode.gm_probseg')],
                ),
                (
                    refmask_wf,
                    outputnode,
                    [
                        ('outputnode.refmask_file', 'refmask'),
                    ],
                ),
                (
                    refmask_wf,
                    ds_refmask_wf,
                    [
                        ('outputnode.refmask_file', 'inputnode.refmask'),
                    ],
                ),
                (
                    inputnode,
                    ds_refmask_wf,
                    [
                        ('segmentation', 'inputnode.segmentation'),
                        ('t1w_preproc', 'inputnode.anat_sources'),
                    ],
                ),
                (
                    gm_select,
                    ds_refmask_wf,
                    [
                        ('out', 'inputnode.source_files'),
                    ],
                ),
                (
                    petref_buffer,
                    refmask_report_wf,
                    [
                        ('pet_file', 'inputnode.source_file'),
                        ('petref', 'inputnode.petref'),
                    ],
                ),
                (
                    refmask_wf,
                    refmask_report_wf,
                    [
                        ('outputnode.refmask_file', 'inputnode.refmask'),
                    ],
                ),
                (
                    refmask_report_wf,
                    func_fit_reports_wf,
                    [
                        ('outputnode.refmask_report', 'inputnode.refmask_report'),
                    ],
                ),
                (
                    outputnode,
                    func_fit_reports_wf,
                    [
                        ('refmask', 'inputnode.refmask'),
                    ],
                ),
                (
                    petref_buffer,
                    pet_ref_tacs_wf,
                    [
                        ('pet_file', 'inputnode.pet_anat'),
                    ],
                ),
                (
                    refmask_wf,
                    pet_ref_tacs_wf,
                    [
                        ('outputnode.refmask_file', 'inputnode.mask_file'),
                    ],
                ),
            ]
        )
        if pvc_method is None:
            workflow.connect(
                [
                    (
                        pet_ref_tacs_wf,
                        ds_ref_tacs,
                        [
                            ('outputnode.timeseries', 'in_file'),
                        ],
                    ),
                ]
            )
    else:
        config.loggers.workflow.info('PET Stage 4: Reference mask generation skipped')

    return workflow


def init_pet_native_wf(
    *,
    pet_series: list[str],
    omp_nthreads: int = 1,
    name: str = 'pet_native_wf',
) -> pe.Workflow:
    r"""
    Minimal resampling workflow.

    This workflow performs slice-timing correction, and resamples to petref space
    with head motion and susceptibility distortion correction. It also selects
    the transforms needed to perform further resampling.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.tests import mock_config
            from petprep import config
            from petprep.workflows.pet.fit import init_pet_native_wf
            with mock_config():
                pet_file = config.execution.bids_dir / "sub-01" / "func" \
                    / "sub-01_task-mixedgamblestask_run-01_pet.nii.gz"
                wf = init_pet_native_wf(pet_series=[str(pet_file)])

    Parameters
    ----------
    pet_series
        List of paths to NIfTI files.

    Inputs
    ------
    petref
        PET reference file
    pet_mask
        Mask of pet reference file
    motion_xfm
        Affine transforms from each PET volume to ``petref``, written
        as concatenated ITK affine transforms.

    Outputs
    -------
    pet_minimal
        PET series ready for further resampling.
    pet_native
        PET series resampled into PET reference space. Head motion correction
        will be applied to each file.
    metadata
        Metadata dictionary of PET series
    motion_xfm
        Motion correction transforms for further correcting pet_minimal.

    """

    layout = config.execution.layout
    pet_series = listify(pet_series)

    all_metadata = [layout.get_metadata(pet_file) for pet_file in pet_series]

    pet_file = pet_series[0]
    metadata = all_metadata[0]

    _, mem_gb = estimate_pet_mem_usage(pet_file)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # PET fit
                'petref',
                'pet_mask',
                'motion_xfm',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_minimal',
                'pet_native',
                'metadata',
                # Transforms
                'motion_xfm',
            ],  # fmt:skip
        ),
        name='outputnode',
    )
    outputnode.inputs.metadata = metadata

    petbuffer = pe.Node(niu.IdentityInterface(fields=['pet_file']), name='petbuffer')

    # PET source: track original PET file(s)
    # The Select interface requires an index to choose from ``inlist``. Since
    # ``pet_file`` is a single path, explicitly set the index to ``0`` to avoid
    # missing mandatory input errors when the node runs.
    pet_source = pe.Node(niu.Select(inlist=pet_series, index=0), name='pet_source')
    validate_pet = pe.Node(ValidateImage(), name='validate_pet')
    workflow.connect([
        (pet_source, validate_pet, [('out', 'in_file')]),
    ])  # fmt:skip

    workflow.connect([(validate_pet, petbuffer, [('out_file', 'pet_file')])])

    # Resample to petref
    petref_pet = pe.Node(
        ResampleSeries(),
        name='petref_pet',
        n_procs=omp_nthreads,
        mem_gb=mem_gb['resampled'],
    )

    workflow.connect([
        (inputnode, petref_pet, [
            ('petref', 'ref_file'),
            ('motion_xfm', 'transforms'),
        ]),
        (petbuffer, petref_pet, [
            ('pet_file', 'in_file'),
        ]),
    ])  # fmt:skip

    workflow.connect([
        (inputnode, outputnode, [('motion_xfm', 'motion_xfm')]),
        (petbuffer, outputnode, [('pet_file', 'pet_minimal')]),
        (petref_pet, outputnode, [('out_file', 'pet_native')]),
    ])  # fmt:skip

    return workflow
