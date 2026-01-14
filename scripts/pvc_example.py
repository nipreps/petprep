#!/usr/bin/env python3
"""
Created on Thu Jun 19 10:16:13 2025

@author: martinnorgaard
"""

from pathlib import Path

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from petprep.workflows.pet_pvc import init_pet_pvc_wf


def test_pet_pvc_workflow(tool='PETPVC', method='GTM'):
    workflow = pe.Workflow(name=f'test_{tool.lower()}_{method.lower()}_workflow')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['pet_file', 'segmentation', 't1w_tpms', 'petref', 'subjects_dir', 'subject_id']
        ),
        name='inputnode',
    )

    pvc_wf = init_pet_pvc_wf(
        tool=tool,
        method=method,
        config_path=Path(
            '/Users/martinnorgaard/Documents/GitHub/petprep/petprep/data/pvc/config.json'
        ),  # Update with your config.json path
        name=f'{tool.lower()}_{method.lower()}_pvc_wf',
    )

    workflow.connect(
        [
            (
                inputnode,
                pvc_wf,
                [
                    ('pet_file', 'inputnode.pet_file'),
                    ('segmentation', 'inputnode.segmentation'),
                    ('t1w_tpms', 'inputnode.t1w_tpms'),
                    ('petref', 'inputnode.petref'),
                    ('subjects_dir', 'inputnode.subjects_dir'),
                    ('subject_id', 'inputnode.subject_id'),
                ],
            )
        ]
    )

    # Define inputs
    workflow.inputs.inputnode.pet_file = '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/sub-010_ses-baseline_space-T1w_desc-preproc_pet.nii.gz'
    workflow.inputs.inputnode.petref = '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/sub-010_ses-baseline_space-T1w_petref.nii.gz'
    workflow.inputs.inputnode.segmentation = '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/sub-010_ses-baseline_desc-gtm_dseg.nii.gz'
    workflow.inputs.inputnode.t1w_tpms = [
        '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/sub-010_ses-baseline_label-GM_probseg.nii.gz',
        '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/sub-010_ses-baseline_label-WM_probseg.nii.gz',
        '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/sub-010_ses-baseline_label-CSF_probseg.nii.gz',
    ]
    workflow.inputs.inputnode.subjects_dir = (
        '/Users/martinnorgaard/Desktop/ses-baseline/test_data/derivatives/freesurfer'
    )
    workflow.inputs.inputnode.subject_id = 'sub-010'

    workflow.base_dir = './workflow_output'
    workflow.run()


# Example usage:
test_pet_pvc_workflow(tool='petsurfer', method='GTM')
