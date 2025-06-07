# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Segmentation workflows."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


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
    """Return a minimal segmentation workflow selecting a FreeSurfer command."""
    name = name or f'pet_{seg}_seg_wf'
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t1w_preproc', 'subjects_dir', 'subject_id']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['segmentation']), name='outputnode')

    # This node is just a placeholder for the actual FreeSurfer command
    seg_node = pe.Node(
        niu.IdentityInterface(fields=['segmentation', 'subjects_dir', 'subject_id']),
        name=f'run_{seg}',
    )

    workflow.connect(
        [
            (inputnode, seg_node, [
                ('subjects_dir', 'subjects_dir'),
                ('subject_id', 'subject_id'),
            ])
        ]
    )

    workflow.connect([(inputnode, seg_node, [])])
    workflow.connect([(seg_node, outputnode, [('segmentation', 'segmentation')])])

    return workflow