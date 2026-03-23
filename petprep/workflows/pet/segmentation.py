# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Segmentation workflows."""

from nipype import Function
from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer import MRIConvert
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import config
from ...data import load as load_data
from ...interfaces import DerivativesDataSink
from ...interfaces.bids import BIDSURI
from ...interfaces.segmentation import (
    MRISclimbicSeg,
    SegmentBS,
    SegmentGTM,
    SegmentHA_T1,
    SegmentThalamicNuclei,
    SegmentWM,
    SegStats,
)
from ...utils.segmentation import (
    ctab_to_dsegtsv,
    gtm_stats_to_stats,
    gtm_to_dsegtsv,
    summary_to_stats,
)

try:  # Py>=3.9
    from importlib.resources import files as ir_files
except Exception:  # pragma: no cover - Py<3.9 fallback
    from importlib_resources import files as ir_files

SEG_DATA = ir_files('petprep.data.segmentation')


def _merge_ha_labels(lh_file: str, rh_file: str) -> str:
    """Combine left and right hippocampus/amygdala label volumes."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    lh_img = nb.load(lh_file)
    rh_img = nb.load(rh_file)

    if not np.allclose(lh_img.affine, rh_img.affine) or lh_img.shape != rh_img.shape:
        raise ValueError('Hemisphere segmentations do not align')

    lh_labels = np.rint(lh_img.get_fdata()).astype(np.int16)
    rh_labels = np.rint(rh_img.get_fdata()).astype(np.int16)
    data = np.where(rh_labels != 0, rh_labels, lh_labels).astype(np.int16)

    out_img = lh_img.__class__(data, lh_img.affine, lh_img.header)
    out_img.set_data_dtype(np.int16)

    out_file = Path('hippocampusAmygdala_dseg.nii.gz').absolute()
    out_img.to_filename(out_file)
    return str(out_file)


SEGMENTATIONS = {
    'gtm': {
        'interface': SegmentGTM,
        'interface_kwargs': {'args': '--xcerseg'},
        'desc': 'gtm',
        'inputs': [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
        'segstats': False,
        'dseg_func': gtm_to_dsegtsv,
        'morph_func': gtm_stats_to_stats,
    },
    'brainstem': {
        'interface': SegmentBS,
        'desc': 'brainstem',
        'inputs': [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
    },
    'thalamicNuclei': {
        'interface': SegmentThalamicNuclei,
        'desc': 'thalamus',
        'inputs': [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
    },
    'hippocampusAmygdala': {
        'interface': SegmentHA_T1,
        'desc': 'hippocampusAmygdala',
        'inputs': [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
        'merge_ha': True,
    },
    'wm': {
        'interface': SegmentWM,
        'desc': 'whiteMatter',
        'inputs': [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
    },
    'raphe': {
        'interface': MRISclimbicSeg,
        'interface_kwargs': {
            'model': str(SEG_DATA / 'raphe+pons.n21.d114.h5'),
            'ctab': str(SEG_DATA / 'raphe+pons.ctab'),
            'out_file': 'raphe_seg.mgz',
            'write_volumes': True,
            'keep_ac': True,
            'percentile': 99.9,
            'vmp': True,
            'conform': True,
        },
        'desc': 'raphe',
        'inputs': [('t1w_preproc', 'in_file')],
        'color_table': str(SEG_DATA / 'raphe+pons_cleaned.ctab'),
    },
    'limbic': {
        'interface': MRISclimbicSeg,
        'interface_kwargs': {
            'ctab': str(load_data('segmentation/sclimbic.ctab')),
            'out_file': 'sclimbic.mgz',
            'write_volumes': True,
            'conform': True,
        },
        'desc': 'limbic',
        'inputs': [('t1w_preproc', 'in_file')],
        'color_table': str(load_data('segmentation/sclimbic_cleaned.ctab')),
    },
}


def _build_nodes(
    seg: str,
    desc: str,
    *,
    color_table: str | None = None,
    segstats: bool = True,
    merge_ha: bool = False,
    dseg_func=ctab_to_dsegtsv,
    morph_func=summary_to_stats,
):
    """Create common segmentation nodes."""
    nodes = {}
    if merge_ha:
        nodes['convert_lh'] = pe.Node(
            MRIConvert(out_type='niigz', resample_type='nearest'), name='convert_ha_lh'
        )
        nodes['convert_rh'] = pe.Node(
            MRIConvert(out_type='niigz', resample_type='nearest'), name='convert_ha_rh'
        )
        nodes['merge_seg'] = pe.Node(
            Function(
                input_names=['lh_file', 'rh_file'],
                output_names=['out_file'],
                function=_merge_ha_labels,
            ),
            name='merge_ha_seg',
        )
        seg_source = nodes['merge_seg']
    else:
        nodes['convert_seg'] = pe.Node(
            MRIConvert(out_type='niigz', resample_type='nearest'), name=f'convert_{seg}seg'
        )
        seg_source = nodes['convert_seg']

    nodes['sources'] = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.petprep_dir),
        ),
        name='sources',
    )

    nodes['ds_seg'] = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            seg=seg,
            allowed_entities=('seg',),
            suffix='dseg',
            extension='.nii.gz',
            compress=True,
        ),
        name=f'ds_{seg}seg',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    if segstats:
        segstats_kwargs = {
            'exclude_id': 0,
            'ctab_out_file': f'desc-{desc}_dseg.ctab',
            'summary_file': f'desc-{desc}_morph.txt',
        }
        if color_table:
            segstats_kwargs['color_table_file'] = color_table
        else:
            segstats_kwargs['default_color_table'] = True
        nodes['segstats'] = pe.Node(SegStats(**segstats_kwargs), name=f'segstats_{seg}')
        nodes['create_morph'] = pe.Node(
            Function(input_names=['summary_file'], output_names=['out_file'], function=morph_func),
            name=f'create_{seg}_morphtsv',
        )
        nodes['create_dseg'] = pe.Node(
            Function(input_names=['ctab_file'], output_names=['out_file'], function=dseg_func),
            name=f'create_{seg}_dsegtsv',
        )
    else:
        nodes['make_dseg'] = pe.Node(
            niu.Function(
                function=dseg_func,
                input_names=['subjects_dir', 'subject_id', 'seg_file'],
                output_names=['out_file'],
            ),
            name=f'make_{seg}dsegtsv',
        )
        nodes['make_morph'] = pe.Node(
            niu.Function(
                function=morph_func,
                input_names=['subjects_dir', 'subject_id', 'seg_file'],
                output_names=['out_file'],
            ),
            name=f'make_{seg}morphtsv',
        )

    nodes['ds_dseg_tsv'] = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            seg=seg,
            allowed_entities=('seg',),
            suffix='dseg',
            extension='.tsv',
            datatype='anat',
            check_hdr=False,
        ),
        name=f'ds_{seg}dsegtsv',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    nodes['ds_morph_tsv'] = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            seg=seg,
            allowed_entities=('seg',),
            suffix='morph',
            extension='.tsv',
            datatype='anat',
            check_hdr=False,
        ),
        name=f'ds_{seg}morphtsv',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    nodes['seg_source'] = seg_source
    return nodes


def init_segmentation_wf(seg: str = 'gtm', name: str | None = None) -> Workflow:
    """Return a minimal segmentation workflow selecting a FreeSurfer command."""
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

    spec = SEGMENTATIONS.get(seg)
    if spec is None:
        seg_node = pe.Node(niu.IdentityInterface(fields=['segmentation']), name=f'run_{seg}')
        workflow.connect([(seg_node, outputnode, [('segmentation', 'segmentation')])])
        return workflow

    interface = spec['interface']
    seg_node = pe.Node(interface(**spec.get('interface_kwargs', {})), name=f'run_{seg}')

    for in_field, out_field in spec.get('inputs', []):
        workflow.connect([(inputnode, seg_node, [(in_field, out_field)])])

    nodes = _build_nodes(
        seg,
        spec['desc'],
        color_table=spec.get('color_table'),
        segstats=spec.get('segstats', True),
        merge_ha=spec.get('merge_ha', False),
        dseg_func=spec.get('dseg_func', ctab_to_dsegtsv),
        morph_func=spec.get('morph_func', summary_to_stats),
    )

    if spec.get('merge_ha', False):
        workflow.connect(
            [
                (seg_node, nodes['convert_lh'], [('lh_hippoAmygLabels', 'in_file')]),
                (seg_node, nodes['convert_rh'], [('rh_hippoAmygLabels', 'in_file')]),
                (inputnode, nodes['convert_lh'], [('t1w_preproc', 'reslice_like')]),
                (inputnode, nodes['convert_rh'], [('t1w_preproc', 'reslice_like')]),
                (nodes['convert_lh'], nodes['merge_seg'], [('out_file', 'lh_file')]),
                (nodes['convert_rh'], nodes['merge_seg'], [('out_file', 'rh_file')]),
            ]
        )
    else:
        workflow.connect(
            [
                (seg_node, nodes['convert_seg'], [('out_file', 'in_file')]),
                (inputnode, nodes['convert_seg'], [('t1w_preproc', 'reslice_like')]),
            ]
        )

    workflow.connect(
        [
            (inputnode, nodes['sources'], [('t1w_preproc', 'in1')]),
            (nodes['seg_source'], nodes['ds_seg'], [('out_file', 'in_file')]),
            (inputnode, nodes['ds_seg'], [('t1w_preproc', 'source_file')]),
            (nodes['sources'], nodes['ds_seg'], [('out', 'Sources')]),
            (nodes['ds_seg'], outputnode, [('out_file', 'segmentation')]),
        ]
    )

    if spec.get('segstats', True):
        workflow.connect(
            [
                (nodes['seg_source'], nodes['segstats'], [('out_file', 'segmentation_file')]),
                (nodes['segstats'], nodes['create_morph'], [('summary_file', 'summary_file')]),
                (nodes['segstats'], nodes['create_dseg'], [('ctab_out_file', 'ctab_file')]),
                (nodes['create_dseg'], nodes['ds_dseg_tsv'], [('out_file', 'in_file')]),
                (nodes['create_morph'], nodes['ds_morph_tsv'], [('out_file', 'in_file')]),
            ]
        )
    else:
        workflow.connect(
            [
                (
                    inputnode,
                    nodes['make_dseg'],
                    [
                        ('subjects_dir', 'subjects_dir'),
                        ('subject_id', 'subject_id'),
                    ],
                ),
                (
                    inputnode,
                    nodes['make_morph'],
                    [
                        ('subjects_dir', 'subjects_dir'),
                        ('subject_id', 'subject_id'),
                    ],
                ),
                (seg_node, nodes['make_dseg'], [('out_file', 'seg_file')]),
                (seg_node, nodes['make_morph'], [('out_file', 'seg_file')]),
                (nodes['make_dseg'], nodes['ds_dseg_tsv'], [('out_file', 'in_file')]),
                (nodes['make_morph'], nodes['ds_morph_tsv'], [('out_file', 'in_file')]),
            ]
        )

    workflow.connect(
        [
            (inputnode, nodes['ds_dseg_tsv'], [('t1w_preproc', 'source_file')]),
            (inputnode, nodes['ds_morph_tsv'], [('t1w_preproc', 'source_file')]),
            (nodes['sources'], nodes['ds_dseg_tsv'], [('out', 'Sources')]),
            (nodes['sources'], nodes['ds_morph_tsv'], [('out', 'Sources')]),
            (nodes['ds_dseg_tsv'], outputnode, [('out_file', 'dseg_tsv')]),
        ]
    )

    return workflow
