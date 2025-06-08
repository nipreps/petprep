# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Workflow to extract time-activity curves."""

from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer import MRIConvert
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink, ExtractTACs
from ...interfaces.bids import BIDSURI
from .outputs import prepare_timing_parameters


def init_gtm_tacs_wf(metadata: dict, name: str = 'gtm_tacs_wf') -> Workflow:
    """Generate TACs from a GTM segmentation."""

    timing_parameters = prepare_timing_parameters(metadata)

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet', 'segmentation', 'dseg_tsv']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.petprep_dir),
        ),
        name='sources',
    )

    resample_seg = pe.Node(
        MRIConvert(out_type='niigz', resample_type='nearest'),
        name='resample_seg',
    )

    extract = pe.Node(
        ExtractTACs(metadata=timing_parameters),
        name='extract_tacs',
    )

    ds_tacs = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            desc='gtm',
            suffix='tacs',
            extension='.tsv',
            datatype='pet',
            check_hdr=False,
        ),
        name='ds_gtmtacs',
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
        [
            (inputnode, sources, [('pet', 'in1')]),
            (
                inputnode,
                resample_seg,
                [
                    ('segmentation', 'in_file'),
                    ('pet', 'reslice_like'),
                ],
            ),
            (
                inputnode,
                extract,
                [
                    ('pet', 'pet_file'),
                    ('dseg_tsv', 'dseg_tsv'),
                ],
            ),
            (resample_seg, extract, [('out_file', 'segmentation')]),
            (extract, ds_tacs, [('out_file', 'in_file')]),
            (inputnode, ds_tacs, [('pet', 'source_file')]),
            (sources, ds_tacs, [('out', 'Sources')]),
            (ds_tacs, outputnode, [('out_file', 'out_file')]),
        ]
    )

    return workflow
