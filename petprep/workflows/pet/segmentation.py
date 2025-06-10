# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Segmentation workflows."""

from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.freesurfer.petsurfer import GTMSeg
from ...interfaces.fs_model import SegStats
from nipype.pipeline import engine as pe
from nipype import Function
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.bids import BIDSURI
from ...interfaces.segmentation import SegmentBS, SegmentThalamicNuclei
from ...utils.brainstem import brainstem_stats_to_stats, brainstem_to_dsegtsv
from ...utils.gtmseg import gtm_stats_to_stats, gtm_to_dsegtsv
from ...utils.thalamic import ctab_to_dsegtsv, summary_to_stats

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
    if seg == 'gtm':
        seg_node = pe.Node(GTMSeg(args='--no-xcerseg'), name='run_gtm')
    elif seg == 'thalamicNuclei':
        seg_node = pe.Node(SegmentThalamicNuclei(), name='run_thalamicnuclei')
    elif seg == 'brainstem':
        seg_node = pe.Node(SegmentBS(), name='run_brainstem')
    else:
        seg_node = pe.Node(niu.IdentityInterface(fields=['segmentation']), name=f'run_{seg}')

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
    elif seg == 'brainstem':
        workflow.connect(
            [
                (
                    inputnode,
                    seg_node,
                    [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
                )
            ]
        )
    elif seg == 'thalamicNuclei':
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
    elif seg == 'thalamicNuclei':
        convert_seg = pe.Node(
            MRIConvert(out_type='niigz', resample_type='nearest'),
            name='convert_thalamicseg',
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
                desc='thalamus',
                suffix='dseg',
                extension='.nii.gz',
                compress=True,
            ),
            name='ds_thalamicseg',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        segstats_thal = pe.Node(
            SegStats(
                exclude_id=0,
                default_color_table=True,
                ctab_out_file='desc-thalamus_dseg.ctab',
                summary_file='desc-thalamus_morph.txt',
                avgwf_txt_file='desc-thalamus_avgwf.txt',
            ),
            name='segstats_thal',
        )

        create_thal_morph = pe.Node(
            Function(
                input_names=['summary_file'],
                output_names=['out_file'],
                function=summary_to_stats,
            ),
            name='create_thal_morphtsv',
        )

        create_thal_dseg = pe.Node(
            Function(
                input_names=['ctab_file'],
                output_names=['out_file'],
                function=ctab_to_dsegtsv,
            ),
            name='create_thal_dsegtsv',
        )

        ds_dseg_tsv = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='thalamus',
                suffix='dseg',
                extension='.tsv',
                datatype='anat',
                check_hdr=False,
            ),
            name='ds_thalamusdsegtsv',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        ds_morph_tsv = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='thalamus',
                suffix='morph',
                extension='.tsv',
                datatype='anat',
                check_hdr=False,
            ),
            name='ds_thalamusmorphtsv',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        workflow.connect(
            [
                (seg_node, convert_seg, [('thalamic_labels_voxel', 'in_file')]),
                (inputnode, convert_seg, [('t1w_preproc', 'reslice_like')]),
                (inputnode, sources, [('t1w_preproc', 'in1')]),
                (convert_seg, ds_seg, [('out_file', 'in_file')]),
                (inputnode, ds_seg, [('t1w_preproc', 'source_file')]),
                (sources, ds_seg, [('out', 'Sources')]),
                (ds_seg, outputnode, [('out_file', 'segmentation')]),
                (seg_node, segstats_thal, [('thalamic_labels_voxel', 'segmentation_file')]),
                (segstats_thal, create_thal_morph, [('summary_file', 'summary_file')]),
                (segstats_thal, create_thal_dseg, [('ctab_out_file', 'ctab_file')]),
                (create_thal_dseg, ds_dseg_tsv, [('out_file', 'in_file')]),
                (create_thal_morph, ds_morph_tsv, [('out_file', 'in_file')]),
                (inputnode, ds_dseg_tsv, [('t1w_preproc', 'source_file')]),
                (inputnode, ds_morph_tsv, [('t1w_preproc', 'source_file')]),
                (sources, ds_dseg_tsv, [('out', 'Sources')]),
                (sources, ds_morph_tsv, [('out', 'Sources')]),
                (ds_dseg_tsv, outputnode, [('out_file', 'dseg_tsv')]),
            ]
        )
    elif seg == 'brainstem':
        convert_seg = pe.Node(
            MRIConvert(out_type='niigz', resample_type='nearest'),
            name='convert_brainstemseg',
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
                desc='brainstem',
                suffix='dseg',
                extension='.nii.gz',
                compress=True,
            ),
            name='ds_brainstemseg',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        segstats_bs = pe.Node(
            SegStats(
                exclude_id=0,
                default_color_table=True,
                ctab_out_file="desc-brainstem_dseg.ctab",
                summary_file="desc-brainstem_morph.txt",
            ),
            name="segstats_bs",
        )

        create_bs_morphtsv = pe.Node(
            Function(
                input_names=["summary_file"],
                output_names=["out_file"],
                function=summary_to_stats,
            ),
            name="create_bs_morphtsv",
        )

        create_bs_dsegtsv = pe.Node(
            Function(
                input_names=["ctab_file"],
                output_names=["out_file"],
                function=ctab_to_dsegtsv,
            ),
            name="create_bs_dsegtsv",
        )

        ds_dseg_tsv = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='brainstem',
                suffix='dseg',
                extension='.tsv',
                datatype='anat',
                check_hdr=False,
            ),
            name='ds_brainstemdsegtsv',
            run_without_submitting=True,
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        ds_morph_tsv = pe.Node(
            DerivativesDataSink(
                base_directory=config.execution.petprep_dir,
                desc='brainstem',
                suffix='morph',
                extension='.tsv',
                datatype='anat',
                check_hdr=False,
            ),
            name='ds_brainstemmorphtsv',
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
                (seg_node, segstats_bs, [('out_file', 'segmentation_file')]),
                (segstats_bs, create_bs_morphtsv, [('summary_file', 'summary_file')]),
                (segstats_bs, create_bs_dsegtsv, [('ctab_out_file', 'ctab_file')]),
                (create_bs_dsegtsv, ds_dseg_tsv, [('out_file', 'in_file')]),
                (create_bs_morphtsv, ds_morph_tsv, [('out_file', 'in_file')]),
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