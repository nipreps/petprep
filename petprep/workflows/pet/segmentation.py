# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Segmentation workflows."""

from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.freesurfer.petsurfer import GTMSeg
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.bids import BIDSURI
from ...utils.gtmseg import gtm_stats_to_stats, gtm_to_dsegtsv

SEGMENTATION_CMDS = {
    'gtm': 'gtmseg',
    'brainstem': 'SegmentBS',
    'thalamicNuclei': 'SegmentThalamicNuclei',
    'hippocampusAmygdala': 'SegmentHA_T1',
    'wm': 'SegmentWM',
    'raphe': 'SegmentBS',
    'limbic': 'SegmentHA_T1',
}


def init_segmentation_wf(seg: str = 'gtm', name: str | None = None) -> Workflow:
    """Return a minimal segmentation workflow selecting a FreeSurfer command.

    When ``seg`` is ``'gtm'``, the workflow runs FreeSurfer's ``gtmseg`` utility.
    In that case, ``subjects_dir`` and ``subject_id`` inputs must be provided.
    """
    name = name or f'pet_{seg}_seg_wf'
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w_preproc', 'subjects_dir', 'subject_id']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['segmentation', 'dseg_tsv']),
        name='outputnode',
    )

    # This node is just a placeholder for the actual FreeSurfer command
    seg_node = (
        pe.Node(GTMSeg(no_xcerseg=True), name=f'run_{seg}')
        if seg == 'gtm'
        else pe.Node(niu.IdentityInterface(fields=['segmentation']), name=f'run_{seg}')
    )

    if seg == 'gtm':
        workflow.connect(
            [
                (
                    inputnode,
                    seg_node,
                    [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
                )
            ]
        )
    else:
        workflow.connect([(inputnode, seg_node, [])])

    if seg == 'gtm':
        convert_seg = pe.Node(
            MRIConvert(out_type='niigz', resample_type='nearest'),
            name='convert_gtmseg',
        )
        sources = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.petprep_dir),
            ),
            name='sources',
        )
        ds_seg = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='gtm',
                suffix='dseg',
                compress=True,
            ),
            name='ds_gtmseg',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        mk_dseg_tsv = pe.Node(
            niu.Function(
                function=gtm_to_dsegtsv,
                input_names=['subjects_dir', 'subject_id'],
                output_names=['out_file'],
            ),
            name='make_gtmdsegtsv',
        )
        mk_morph_tsv = pe.Node(
            niu.Function(
                function=gtm_stats_to_stats,
                input_names=['subjects_dir', 'subject_id'],
                output_names=['out_file'],
            ),
            name='make_gtmmorphtsv',
        )
        ds_dseg_tsv = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='gtm',
                suffix='dseg',
                extension='.tsv',
                datatype='anat',
                check_hdr=False,
            ),
            name='ds_gtmdsegtsv',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        ds_morph_tsv = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='gtm',
                suffix='morph',
                extension='.tsv',
                datatype='anat',
                check_hdr=False,
            ),
            name='ds_gtmmorphtsv',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        workflow.connect(
            [
                (seg_node, convert_seg, [('out_file', 'in_file')]),
                (inputnode, convert_seg, [('t1w_preproc', 'reslice_like')]),
                (inputnode, sources, [('t1w_preproc', 'in1')]),
                (convert_seg, ds_seg, [('out_file', 'in_file')]),
                (inputnode, ds_seg, [('t1w_preproc', 'source_file')]),
                (sources, ds_seg, [('out', 'Sources')]),
                (ds_seg, outputnode, [('out_file', 'segmentation')]),
                (
                    inputnode,
                    mk_dseg_tsv,
                    [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
                ),
                (
                    inputnode,
                    mk_morph_tsv,
                    [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
                ),
                (mk_dseg_tsv, ds_dseg_tsv, [('out_file', 'in_file')]),
                (mk_morph_tsv, ds_morph_tsv, [('out_file', 'in_file')]),
                (inputnode, ds_dseg_tsv, [('t1w_preproc', 'source_file')]),
                (inputnode, ds_morph_tsv, [('t1w_preproc', 'source_file')]),
                (sources, ds_dseg_tsv, [('out', 'Sources')]),
                (sources, ds_morph_tsv, [('out', 'Sources')]),
                (ds_dseg_tsv, outputnode, [('out_file', 'dseg_tsv')]),
            ]
        )
    else:
        workflow.connect([(seg_node, outputnode, [('segmentation', 'segmentation')])])

    return workflow
