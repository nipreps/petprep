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
"""Writing out derivative files."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from niworkflows.utils.images import dseg_label

from petprep import config
from petprep.config import DEFAULT_MEMORY_MIN_GB
from petprep.interfaces import DerivativesDataSink
from petprep.interfaces.bids import BIDSURI
from petprep.interfaces.maths import CropAroundMask


def prepare_timing_parameters(metadata: dict):
    """Convert initial timing metadata to derivative timing parameters.

    Slice timing information is ignored and outputs will always indicate that
    slice timing correction was not performed.

    Examples
    --------

    >>> prepare_timing_parameters({'FrameTimesStart': [0, 2, 6], 'FrameDuration': [2, 4, 4]})
    {'FrameTimesStart': [0, 2, 6], 'FrameDuration': [2, 4, 4]}
    """
    timing_parameters = {}

    frame_times = metadata.get('FrameTimesStart') or metadata.get('VolumeTiming')
    frame_duration = metadata.get('FrameDuration') or metadata.get('AcquisitionDuration')

    if frame_times is not None:
        timing_parameters['FrameTimesStart'] = frame_times

    if frame_duration is not None:
        timing_parameters['FrameDuration'] = frame_duration

    for key in ('InjectedRadioactivity', 'InjectedRadioactivityUnits', 'Units'):
        if key in metadata:
            timing_parameters[key] = metadata[key]

    return timing_parameters


def build_psf_dict(fwhm_x=None, fwhm_y=None, fwhm_z=None):
    """Construct a metadata dictionary for PSF parameters."""
    from nipype.interfaces.base import Undefined as _Undefined

    if (
        fwhm_x is None
        or fwhm_y is None
        or fwhm_z is None
        or fwhm_x is _Undefined
        or fwhm_y is _Undefined
        or fwhm_z is _Undefined
    ):
        return {}
    return {
        'fwhm_x': float(fwhm_x),
        'fwhm_y': float(fwhm_y),
        'fwhm_z': float(fwhm_z),
    }


def init_func_fit_reports_wf(
    *,
    freesurfer: bool,
    output_dir: str,
    ref_name: str,
    name='func_fit_reports_wf',
) -> pe.Workflow:
    """
    Set up a battery of datasinks to store reports in the right location.

    Parameters
    ----------
    freesurfer : :obj:`bool`
        FreeSurfer was enabled
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: anat_reports_wf)

    Inputs
    ------
    source_file
        Input PET images

    std_t1w
        T1w image resampled to standard space
    std_mask
        Mask of skull-stripped template
    subject_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    t1w_conform_report
        Conformation report
    t1w_preproc
        The T1w reference map, which is calculated as the average of bias-corrected
        and preprocessed T1w images, defining the anatomical space.
    t1w_dseg
        Segmentation in T1w space
    t1w_mask
        Brain (binary) mask estimated by brain extraction.
    template
        Template space and specifications
    summary_report
        Summary of preprocessing steps
    validation_report
        Reportlet from input data validation
    ref_name
        Name of the reference region mask
    refmask_report
        Reportlet showing the reference region mask

    """
    from nireports.interfaces.reporting.base import (
        SimpleBeforeAfterRPT as SimpleBeforeAfter,
    )

    workflow = pe.Workflow(name=name)

    inputfields = [
        'source_file',
        'petref',
        'report_pet',
        'pet_mask',
        'petref2anat_xfm',
        't1w_preproc',
        't1w_mask',
        't1w_dseg',
        'refmask',
        # May be missing
        'subject_id',
        'subjects_dir',
        # Report snippets
        'summary_report',
        'validation_report',
    ]
    if ref_name:
        inputfields.append('refmask_report')
    inputnode = pe.Node(niu.IdentityInterface(fields=inputfields), name='inputnode')

    ds_summary = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='summary',
            datatype='figures',
        ),
        name='ds_report_summary',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    ds_validation = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='validation',
            datatype='figures',
        ),
        name='ds_report_validation',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    if ref_name:
        ds_refmask_report = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='ref',
                label=ref_name,
                datatype='figures',
                allowed_entities=('label',),
            ),
            name='ds_report_refmask',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

    # Resample anatomical references into PET space for plotting
    t1w_petref = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            float=True,
            invert_transform_flags=[True],
            interpolation='LanczosWindowedSinc',
        ),
        name='t1w_petref',
        mem_gb=1,
    )

    t1w_wm = pe.Node(
        niu.Function(function=dseg_label),
        name='t1w_wm',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    t1w_wm.inputs.label = 2  # BIDS default is WM=2

    petref_wm = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            invert_transform_flags=[True],
            interpolation='NearestNeighbor',
        ),
        name='petref_wm',
        mem_gb=1,
    )

    crop_petref = pe.Node(CropAroundMask(), name='crop_petref', mem_gb=0.1)
    crop_t1w_petref = pe.Node(CropAroundMask(), name='crop_t1w_petref', mem_gb=0.1)
    crop_petref_wm = pe.Node(CropAroundMask(), name='crop_petref_wm', mem_gb=0.1)

    if ref_name:
        petref_refmask = pe.Node(
            ApplyTransforms(
                dimension=3,
                default_value=0,
                invert_transform_flags=[True],
                interpolation='NearestNeighbor',
            ),
            name='petref_refmask',
            mem_gb=1,
        )
        crop_petref_refmask = pe.Node(CropAroundMask(), name='crop_petref_refmask', mem_gb=0.1)

    # fmt:off
    workflow.connect([
        (inputnode, ds_summary, [
            ('source_file', 'source_file'),
            ('summary_report', 'in_file'),
        ]),
        (inputnode, ds_validation, [
            ('source_file', 'source_file'),
            ('validation_report', 'in_file'),
        ]),
        (inputnode, t1w_petref, [
            ('t1w_preproc', 'input_image'),
            ('report_pet', 'reference_image'),
            ('petref2anat_xfm', 'transforms'),
        ]),
        (inputnode, t1w_wm, [('t1w_dseg', 'in_seg')]),
        (inputnode, petref_wm, [
            ('report_pet', 'reference_image'),
            ('petref2anat_xfm', 'transforms'),
        ]),
        (t1w_wm, petref_wm, [('out', 'input_image')]),
        (inputnode, crop_petref, [('report_pet', 'in_file'), ('pet_mask', 'mask_file')]),
        (t1w_petref, crop_t1w_petref, [('output_image', 'in_file')]),
        (inputnode, crop_t1w_petref, [('pet_mask', 'mask_file')]),
        (petref_wm, crop_petref_wm, [('output_image', 'in_file')]),
        (inputnode, crop_petref_wm, [('pet_mask', 'mask_file')]),
    ])
    if ref_name:
        workflow.connect([
            (inputnode, ds_refmask_report, [
                ('source_file', 'source_file'),
                ('refmask_report', 'in_file'),
            ]),
            (inputnode, petref_refmask, [
                ('refmask', 'input_image'),
                ('report_pet', 'reference_image'),
                ('petref2anat_xfm', 'transforms'),
            ]),
            (petref_refmask, crop_petref_refmask, [('output_image', 'in_file')]),
            (inputnode, crop_petref_refmask, [('pet_mask', 'mask_file')]),
        ])
    # fmt:on

    # EPI-T1 registration
    # Resample T1w image onto EPI-space

    pet_t1_report = pe.Node(
        SimpleBeforeAfter(
            before_label='T1w',
            after_label='PET',
            dismiss_affine=True,
        ),
        name='pet_t1_report',
        mem_gb=0.1,
    )

    ds_pet_t1_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='coreg',
            suffix='pet',
            datatype='figures',
        ),
        name='ds_pet_t1_report',
    )

    if ref_name:
        pet_t1_refmask_report = pe.Node(
            SimpleBeforeAfter(
                before_label='T1w',
                after_label='PET',
                dismiss_affine=True,
            ),
            name='pet_t1_refmask_report',
            mem_gb=0.1,
        )

        ds_pet_t1_refmask_report = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='ref',
                label=ref_name,
                suffix='pet',
                datatype='figures',
                allowed_entities=('label',),
            ),
            name='ds_pet_t1_refmask_report',
        )

    # fmt:off
    workflow.connect([
        (crop_petref, pet_t1_report, [('out_file', 'after')]),
        (crop_t1w_petref, pet_t1_report, [('out_file', 'before')]),
        (crop_petref_wm, pet_t1_report, [('out_file', 'wm_seg')]),
        (inputnode, ds_pet_t1_report, [('source_file', 'source_file')]),
        (pet_t1_report, ds_pet_t1_report, [('out_report', 'in_file')]),
    ])
    if ref_name:
        workflow.connect([
            (crop_petref, pet_t1_refmask_report, [('out_file', 'after')]),
            (crop_t1w_petref, pet_t1_refmask_report, [('out_file', 'before')]),
            (crop_petref_refmask, pet_t1_refmask_report, [('out_file', 'wm_seg')]),
            (inputnode, ds_pet_t1_refmask_report, [('source_file', 'source_file')]),
            (pet_t1_refmask_report, ds_pet_t1_refmask_report, [('out_report', 'in_file')]),
        ])
    # fmt:on

    return workflow


__all__ = (
    'prepare_timing_parameters',
    'init_func_fit_reports_wf',
    'init_ds_petref_wf',
    'init_ds_petmask_wf',
    'init_ds_refmask_wf',
    'init_ds_registration_wf',
    'init_ds_hmc_wf',
    'init_ds_pet_native_wf',
    'init_ds_volumes_wf',
    'init_pet_preproc_report_wf',
    'init_refmask_report_wf',
)


def init_ds_petref_wf(
    *,
    bids_root,
    output_dir,
    desc: str,
    name='ds_petref_wf',
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'petref']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['petref']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_petref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc=desc,
            datatype='pet',
            suffix='petref',
            compress=True,
        ),
        name='ds_petref',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_petref, [('petref', 'in_file'),
                                 ('source_files', 'source_file')]),
        (sources, ds_petref, [('out', 'Sources')]),
        (ds_petref, outputnode, [('out_file', 'petref')]),
    ])
    # fmt:on

    return workflow


def init_ds_petmask_wf(
    *,
    output_dir,
    desc: str,
    name='ds_petmask_wf',
) -> pe.Workflow:
    """Write out a PET mask."""
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'petmask']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['petmask']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_petmask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc=desc,
            datatype='pet',
            suffix='mask',
            compress=True,
        ),
        name='ds_petmask',
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_petmask, [
            ('petmask', 'in_file'),
            ('source_files', 'source_file'),
        ]),
        (sources, ds_petmask, [('out', 'Sources')]),
        (ds_petmask, outputnode, [('out_file', 'petmask')]),
    ])  # fmt:skip

    return workflow


def init_ds_refmask_wf(
    *,
    output_dir,
    ref_name: str,
    name='ds_refmask_wf',
) -> pe.Workflow:
    """Write out a reference region mask.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    ref_name : :obj:`str`
        Name of the reference region mask
    name : :obj:`str`, optional
        Workflow name (default: ``ds_refmask_wf``)
    """

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'anat_sources', 'segmentation', 'refmask']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['refmask']), name='outputnode')

    merge_source_files = pe.Node(niu.Merge(3), name='merge_source_files')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_refmask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            datatype='anat',
            suffix='mask',
            desc='ref',
            label=ref_name,
            allowed_entities=('label',),
            compress=True,
        ),
        name='ds_refmask',
        run_without_submitting=True,
    )

    workflow.connect([
        (inputnode, merge_source_files, [
            ('source_files', 'in1'),
            ('segmentation', 'in2'),
            ('anat_sources', 'in3'),
        ]),
        (merge_source_files, sources, [('out', 'in1')]),
        (inputnode, ds_refmask, [
            ('refmask', 'in_file'),
        ]),
        (merge_source_files, ds_refmask, [('out', 'source_file')]),
        (sources, ds_refmask, [('out', 'Sources')]),
        (ds_refmask, outputnode, [('out_file', 'refmask')]),
    ])  # fmt:skip

    return workflow


def init_ds_registration_wf(
    *,
    bids_root: str,
    output_dir: str,
    source: str,
    dest: str,
    name: str,
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'xform']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['xform']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_xform = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            mode='image',
            suffix='xfm',
            extension='.txt',
            **{'from': source, 'to': dest},
        ),
        name='ds_xform',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # fmt:off
    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_xform, [('xform', 'in_file'),
                               ('source_files', 'source_file')]),
        (sources, ds_xform, [('out', 'Sources')]),
        (ds_xform, outputnode, [('out_file', 'xform')]),
    ])
    # fmt:on

    return workflow


def init_ds_hmc_wf(
    *,
    bids_root,
    output_dir,
    name='ds_hmc_wf',
) -> pe.Workflow:
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_files', 'xforms']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['xforms']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )

    ds_xforms = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='hmc',
            suffix='xfm',
            extension='.txt',
            compress=True,
            **{'from': 'orig', 'to': 'petref'},
        ),
        name='ds_xforms',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, sources, [('source_files', 'in1')]),
        (inputnode, ds_xforms, [('xforms', 'in_file'),
                                ('source_files', 'source_file')]),
        (sources, ds_xforms, [('out', 'Sources')]),
        (ds_xforms, outputnode, [('out_file', 'xforms')]),
    ])
    # fmt:on

    return workflow


def init_ds_pet_native_wf(
    *,
    bids_root: str,
    output_dir: str,
    pet_output: bool,
    all_metadata: list[dict],
    name='ds_pet_native_wf',
) -> pe.Workflow:
    metadata = all_metadata[0]
    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_files',
                'pet',
                # Transforms previously used to generate the outputs
                'motion_xfm',
            ]
        ),
        name='inputnode',
    )

    sources = pe.Node(
        BIDSURI(
            numinputs=3,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )
    workflow.connect([
        (inputnode, sources, [
            ('source_files', 'in1'),
            ('motion_xfm', 'in2'),
        ]),
    ])  # fmt:skip

    if pet_output:
        ds_pet = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc='preproc',
                datatype='pet',
                compress=True,
                TaskName=metadata.get('TaskName'),
                **timing_parameters,
            ),
            name='ds_pet',
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_pet, [
                ('source_files', 'source_file'),
                ('pet', 'in_file'),
            ]),
            (sources, ds_pet, [('out', 'Sources')]),
        ])  # fmt:skip

    return workflow


def init_ds_volumes_wf(
    *,
    bids_root: str,
    output_dir: str,
    metadata: list[dict],
    pvc_method: str | None = None,
    name='ds_volumes_wf',
) -> pe.Workflow:
    timing_parameters = prepare_timing_parameters(metadata)

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'source_files',
                'ref_file',
                'pet',  # Resampled into target space
                'pet_mask',  # petref space
                'pet_ref',  # petref space
                't2star',  # petref space
                'template',  # target reference image from original transform
                # Anatomical
                'petref2anat_xfm',
                # Template
                'anat2std_xfm',
                # Entities
                'space',
                'cohort',
                'resolution',
                # Transforms previously used to generate the outputs
                'motion_xfm',
                # PSF parameters from PVC
                'fwhm_x',
                'fwhm_y',
                'fwhm_z',
            ]
        ),
        name='inputnode',
    )

    sources = pe.Node(
        BIDSURI(
            numinputs=6,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )
    petref2target = pe.Node(niu.Merge(2), name='petref2target')

    psf_meta = pe.Node(
        niu.Function(
            input_names=['fwhm_x', 'fwhm_y', 'fwhm_z'],
            output_names=['meta_dict'],
            function=build_psf_dict,
        ),
        name='psf_meta',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    # PET is pre-resampled
    ds_pet = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='preproc',
            datatype='pet',
            compress=True,
            TaskName=metadata.get('TaskName'),
            **timing_parameters,
        ),
        name='ds_pet',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    if pvc_method is not None:
        ds_pet.inputs.pvc = pvc_method
    workflow.connect([
        (inputnode, sources, [
            ('source_files', 'in1'),
            ('motion_xfm', 'in2'),
            ('petref2anat_xfm', 'in4'),
            ('anat2std_xfm', 'in5'),
            ('template', 'in6'),
        ]),
        (inputnode, petref2target, [
            # Note that ANTs expects transforms in target-to-source order
            # Reverse this for nitransforms-based resamplers
            ('anat2std_xfm', 'in1'),
            ('petref2anat_xfm', 'in2'),
        ]),
        (inputnode, ds_pet, [
            ('source_files', 'source_file'),
            ('pet', 'in_file'),
            ('space', 'space'),
            ('cohort', 'cohort'),
            ('resolution', 'resolution'),
        ]),
        (sources, ds_pet, [('out', 'Sources')]),
        (psf_meta, ds_pet, [('meta_dict', 'meta_dict')]),
    ])  # fmt:skip

    resample_ref = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            float=True,
            interpolation='LanczosWindowedSinc',
        ),
        name='resample_ref',
    )
    resample_mask = pe.Node(ApplyTransforms(interpolation='MultiLabel'), name='resample_mask')
    resamplers = [resample_ref, resample_mask]

    workflow.connect([
        (inputnode, resample_ref, [('pet_ref', 'input_image')]),
        (inputnode, resample_mask, [('pet_mask', 'input_image')]),
    ])  # fmt:skip

    ds_ref = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            datatype='pet',
            suffix='petref',
            compress=True,
        ),
        name='ds_ref',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    ds_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='brain',
            datatype='pet',
            suffix='mask',
            compress=True,
        ),
        name='ds_mask',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )
    datasinks = [ds_ref, ds_mask]

    workflow.connect(
        [
            (
                inputnode,
                psf_meta,
                [
                    ('fwhm_x', 'fwhm_x'),
                    ('fwhm_y', 'fwhm_y'),
                    ('fwhm_z', 'fwhm_z'),
                ],
            )
        ]
    )

    workflow.connect(
        [
            (inputnode, resampler, [('ref_file', 'reference_image')])
            for resampler in resamplers
        ] + [
            (petref2target, resampler, [('out', 'transforms')])
            for resampler in resamplers
        ] + [
            (inputnode, datasink, [
                ('source_files', 'source_file'),
                ('space', 'space'),
                ('cohort', 'cohort'),
                ('resolution', 'resolution'),
            ])
            for datasink in datasinks
        ] + [
            (sources, datasink, [('out', 'Sources')])
            for datasink in datasinks
        ] + [
            (resampler, datasink, [('output_image', 'in_file')])
            for resampler, datasink in zip(resamplers, datasinks, strict=False)
        ] + [
            (psf_meta, datasink, [('meta_dict', 'meta_dict')])
            for datasink in datasinks
        ]
    )  # fmt:skip

    return workflow


def init_pet_preproc_report_wf(
    mem_gb: float,
    reportlets_dir: str,
    name: str = 'pet_preproc_report_wf',
):
    """
    Generate a visual report.

    This workflow generates and saves a reportlet showing the effect of resampling
    the PET signal using the standard deviation maps.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.pet.resampling import init_pet_preproc_report_wf
            wf = init_pet_preproc_report_wf(mem_gb=1, reportlets_dir='.')

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of PET file in GB
    reportlets_dir : :obj:`str`
        Directory in which to save reportlets
    name : :obj:`str`, optional
        Workflow name (default: pet_preproc_report_wf)

    Inputs
    ------
    in_pre
        PET time-series, before resampling
    in_post
        PET time-series, after resampling
    name_source
        PET series NIfTI file
        Used to recover original information lost during processing

    """
    from nipype.algorithms.confounds import TSNR
    from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    from ...interfaces import DerivativesDataSink

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['in_pre', 'in_post', 'name_source']), name='inputnode'
    )

    pre_tsnr = pe.Node(TSNR(), name='pre_tsnr', mem_gb=mem_gb * 4.5)
    pos_tsnr = pe.Node(TSNR(), name='pos_tsnr', mem_gb=mem_gb * 4.5)

    pet_rpt = pe.Node(SimpleBeforeAfterRPT(), name='pet_rpt', mem_gb=0.1)
    ds_report_pet = pe.Node(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc='preproc',
            datatype='figures',
        ),
        name='ds_report_pet',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    # fmt:off
    workflow.connect([
        (inputnode, ds_report_pet, [('name_source', 'source_file')]),
        (inputnode, pre_tsnr, [('in_pre', 'in_file')]),
        (inputnode, pos_tsnr, [('in_post', 'in_file')]),
        (pre_tsnr, pet_rpt, [('stddev_file', 'before')]),
        (pos_tsnr, pet_rpt, [('stddev_file', 'after')]),
        (pet_rpt, ds_report_pet, [('out_report', 'in_file')]),
    ])
    # fmt:on

    return workflow


def init_refmask_report_wf(
    *, output_dir: str, ref_name: str, name: str = 'refmask_report_wf'
) -> pe.Workflow:
    """Generate a reportlet for the reference mask.

    Parameters
    ----------
    output_dir : :obj:`str`
        Directory in which to save derivatives
    ref_name : :obj:`str`
        Name of the reference region mask
    name : :obj:`str`, optional
        Workflow name (default: ``refmask_report_wf``)
    """

    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['source_file', 'petref', 'refmask']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['refmask_report']), name='outputnode')

    mask_report = pe.Node(SimpleShowMaskRPT(), name='mask_report')
    ds_mask_report = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc='ref',
            label=ref_name,
            datatype='figures',
            allowed_entities=('label',),
            suffix='pet',
        ),
        name='ds_report_refmask',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
        [
            (inputnode, mask_report, [('petref', 'background_file'), ('refmask', 'mask_file')]),
            (inputnode, ds_mask_report, [('source_file', 'source_file')]),
            (mask_report, ds_mask_report, [('out_report', 'in_file')]),
            (mask_report, outputnode, [('out_report', 'refmask_report')]),
        ]
    )

    return workflow
