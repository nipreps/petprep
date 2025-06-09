# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Workflow to extract time-activity curves."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer.petsurfer import GTMPVC
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink, ExtractTACs
from ...interfaces.bids import BIDSURI
from .outputs import prepare_timing_parameters


def _make_mask(segmentation: str, labels: list[int]) -> str:
    """Generate a binary mask from ``segmentation`` selecting ``labels``."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)
    data = np.asanyarray(img.dataobj).astype(int)
    mask = np.isin(data, np.asarray(labels, dtype=int))
    out_file = Path('mask.nii.gz').absolute()
    out_img = img.__class__(mask.astype('uint8'), img.affine, img.header)
    out_img.set_data_dtype('uint8')
    out_img.to_filename(out_file)
    return str(out_file)


def _extract_mask_tacs(
    pet_file: str, mask_file: str, metadata: dict | None, name: str
) -> str:
    """Extract mean TACs from ``pet_file`` within ``mask_file``."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np
    import pandas as pd

    pet_img = nb.load(pet_file)
    data = np.asanyarray(pet_img.dataobj)
    if data.ndim == 3:
        data = data[..., np.newaxis]

    mask = nb.load(mask_file).get_fdata() > 0
    flat_pet = data.reshape(-1, data.shape[3])
    flat_mask = mask.reshape(-1)

    vals = flat_pet[flat_mask]
    if vals.size:
        series = vals.mean(axis=0)
    else:
        series = [np.nan] * data.shape[3]

    meta = metadata or {}
    frame_starts = meta.get('FrameTimesStart')
    frame_durs = meta.get('FrameDuration')

    if not isinstance(frame_starts, list | tuple) and frame_starts is not None:
        frame_starts = [frame_starts]
    if not isinstance(frame_durs, list | tuple) and frame_durs is not None:
        frame_durs = [frame_durs]

    if frame_starts is None:
        frame_starts = [0] if data.shape[3] == 1 else list(range(data.shape[3]))
    if frame_durs is None:
        frame_durs = [1] * data.shape[3]

    frame_starts = list(frame_starts)[: data.shape[3]]
    frame_durs = list(frame_durs)[: data.shape[3]]
    frame_ends = [s + d for s, d in zip(frame_starts, frame_durs, strict=False)]

    out_df = pd.DataFrame({
        'FrameTimeStart': frame_starts,
        'FrameTimesEnd': frame_ends,
        name: series,
    })

    out_file = Path(f'{name}_tacs.tsv').absolute()
    out_df.to_csv(out_file, sep='\t', index=False, float_format='%.8g')
    return str(out_file)


def init_tacs_wf(
    metadata: dict,
    seg_name: str,
    desc_suffix: str,
    ref_labels: list[int] | None = None,
    hb_labels: list[int] | None = None,
    name: str | None = None,
) -> Workflow:
    """Extract TACs from ``segmentation`` in PET space."""
    timing_parameters = prepare_timing_parameters(metadata)

    name = name or f'{seg_name}_tacs_wf'
    workflow = Workflow(name=name)

    input_fields = ['pet', 'segmentation', 'dseg_tsv', 'reg_lta']

    inputnode = pe.Node(
        niu.IdentityInterface(fields=input_fields),
        name='inputnode',
    )
    pvc_method = getattr(config.workflow, 'pvc_method', 'none')

    out_fields = ['out_file']
    if pvc_method != 'none':
        out_fields.append('pvc_tacs')
    if ref_labels:
        out_fields += ['ref_mask', 'ref_tacs']
    if hb_labels:
        out_fields += ['hb_mask', 'hb_tacs']
    outputnode = pe.Node(niu.IdentityInterface(fields=out_fields), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.petprep_dir),
        ),
        name='sources',
    )

    gtmpvc = pe.Node(GTMPVC(no_pvc=True), name='gtmpvc')
    if getattr(config.workflow, 'pvc_psf', None):
        psf = config.workflow.pvc_psf
        if isinstance(psf, (list | tuple)) and len(psf) == 3:
            gtmpvc.inputs.psf_col, gtmpvc.inputs.psf_row, gtmpvc.inputs.psf_slice = psf
        else:
            gtmpvc.inputs.psf = psf

    extract_nopvc_tacs = pe.Node(
        ExtractTACs(metadata=timing_parameters),
        name="extract_nopvc_tacs",
    )

    if pvc_method != 'none':
        extract_pvc_tacs = pe.Node(
            ExtractTACs(metadata=timing_parameters),
            name="extract_pvc_tacs",
        )

    ds_tacs = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            desc=seg_name,
            suffix="tacs",
            extension=".tsv",
            datatype="pet",
            check_hdr=False,
        ),
        name="ds_gtmtacs",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    if pvc_method != 'none':
        pvc_desc = f"{seg_name}_pvc-{pvc_method}"
        ds_pvc_tacs = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc=pvc_desc,
                suffix="tacs",
                extension=".tsv",
                datatype="pet",
                check_hdr=False,
            ),
            name="ds_gtmpvctacs",
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

    if ref_labels:
        ref_mask = pe.Node(
            niu.Function(
                function=_make_mask,
                input_names=['segmentation', 'labels'],
                output_names=['out_file'],
            ),
            name='ref_mask',
        )
        ref_mask.inputs.labels = ref_labels
        ref_tacs = pe.Node(
            niu.Function(
                function=_extract_mask_tacs,
                input_names=['pet_file', 'mask_file', 'metadata', 'name'],
                output_names=['out_file'],
            ),
            name='ref_tacs',
        )
        ref_tacs.inputs.metadata = timing_parameters
        ref_tacs.inputs.name = 'ref'
        ds_ref_mask = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='ref',
                suffix='mask',
                compress=True,
            ),
            name='ds_ref_mask',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_ref_tacs = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='ref',
                suffix='tacs',
                extension='.tsv',
                datatype='pet',
                check_hdr=False,
            ),
            name='ds_ref_tacs',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

    if hb_labels:
        hb_mask = pe.Node(
            niu.Function(
                function=_make_mask,
                input_names=['segmentation', 'labels'],
                output_names=['out_file'],
            ),
            name='hb_mask',
        )
        hb_mask.inputs.labels = hb_labels
        hb_tacs = pe.Node(
            niu.Function(
                function=_extract_mask_tacs,
                input_names=['pet_file', 'mask_file', 'metadata', 'name'],
                output_names=['out_file'],
            ),
            name='hb_tacs',
        )
        hb_tacs.inputs.metadata = timing_parameters
        hb_tacs.inputs.name = 'hb'
        ds_hb_mask = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='hb',
                suffix='mask',
                compress=True,
            ),
            name='ds_hb_mask',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )
        ds_hb_tacs = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='hb',
                suffix='tacs',
                extension='.tsv',
                datatype='pet',
                check_hdr=False,
            ),
            name='ds_hb_tacs',
            run_without_submitting=True,
            mem_gb=DEFAULT_MEMORY_MIN_GB,
        )

    workflow.connect(
        [
            (inputnode, sources, [('pet', 'in1')]),
            (
                inputnode,
                gtmpvc,
                [
                    ('pet', 'in_file'),
                    ('segmentation', 'segmentation'),
                    ('reg_lta', 'reg_file'),
                ],
            ),
            (
                gtmpvc,
                extract_nopvc_tacs,
                [("nopvc_file", "pet_file"), ("seg", "segmentation")],
            ),
            (inputnode, extract_nopvc_tacs, [("dseg_tsv", "dseg_tsv")]),
            (extract_nopvc_tacs, ds_tacs, [("out_file", "in_file")]),
            (inputnode, ds_tacs, [("pet", "source_file")]),
            (sources, ds_tacs, [("out", "Sources")]),
            (ds_tacs, outputnode, [("out_file", "out_file")]),
        ]
    )

    if pvc_method != 'none':
        workflow.connect(
            [
                (
                    gtmpvc,
                    extract_pvc_tacs,
                    [("gtm_file", "pet_file"), ("seg", "segmentation")],
                ),
                (inputnode, extract_pvc_tacs, [("dseg_tsv", "dseg_tsv")]),
                (extract_pvc_tacs, ds_pvc_tacs, [("out_file", "in_file")]),
                (inputnode, ds_pvc_tacs, [("pet", "source_file")]),
                (sources, ds_pvc_tacs, [("out", "Sources")]),
                (ds_pvc_tacs, outputnode, [("out_file", "pvc_tacs")]),
            ]
        )

    if ref_labels:
        workflow.connect(
            [
                (inputnode, ref_mask, [("segmentation", "segmentation")]),
                (ref_mask, ref_tacs, [("out_file", "mask_file")]),
                (ref_mask, ds_ref_mask, [("out_file", "in_file")]),
                (inputnode, ds_ref_mask, [("pet", "source_file")]),
                (sources, ds_ref_mask, [("out", "Sources")]),
                (ref_tacs, ds_ref_tacs, [("out_file", "in_file")]),
                (inputnode, ds_ref_tacs, [("pet", "source_file")]),
                (sources, ds_ref_tacs, [("out", "Sources")]),
                (ds_ref_mask, outputnode, [("out_file", "ref_mask")]),
                (ds_ref_tacs, outputnode, [("out_file", "ref_tacs")]),
            ]
        )

    if hb_labels:
        workflow.connect(
            [
                (inputnode, hb_mask, [("segmentation", "segmentation")]),
                (hb_mask, hb_tacs, [("out_file", "mask_file")]),
                (hb_mask, ds_hb_mask, [("out_file", "in_file")]),
                (inputnode, ds_hb_mask, [("pet", "source_file")]),
                (sources, ds_hb_mask, [("out", "Sources")]),
                (hb_tacs, ds_hb_tacs, [("out_file", "in_file")]),
                (inputnode, ds_hb_tacs, [("pet", "source_file")]),
                (sources, ds_hb_tacs, [("out", "Sources")]),
                (ds_hb_mask, outputnode, [("out_file", "hb_mask")]),
                (ds_hb_tacs, outputnode, [("out_file", "hb_tacs")]),
            ]
        )

    return workflow
