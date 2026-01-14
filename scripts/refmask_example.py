#!/usr/bin/env python3
"""
Created on Thu Jun 26 15:52:39 2025

@author: martinnorgaard
"""

# Created on Tue Jun 24 09:45:55 2025
# @author: martinnorgaard

from pathlib import Path

from petprep.workflows.reference_mask import init_pet_refmask_wf

# Test input paths
seg_file = (
    '/Users/martinnorgaard/Dropbox/Mac/Desktop/ses-baseline/test_pvc/'
    'sub-010_ses-baseline_desc-gtm_dseg.nii.gz'
)  # <-- update this path
config_file = (
    '/Users/martinnorgaard/Dropbox/Mac/Documents/GitHub/'
    'petprep/petprep/data/reference_mask/config.json'
)  # <-- update this path
output_dir = Path('test_refmask_output')  # output directory
output_dir.mkdir(exist_ok=True)

# Workflow configuration
segmentation = 'gtm'  # matches top-level key in config.json
ref_mask_name = 'cerebellum'  # matches inner key under segmentation
ref_mask_index = [8, 47]  # OR: ref_mask_index = [3, 42]

# Initialize workflow
wf = init_pet_refmask_wf(
    segmentation=segmentation,
    ref_mask_name=ref_mask_name,
    ref_mask_index=ref_mask_index,
    config_path=config_file,
    name='test_refmask_wf',
)

# Provide the segmentation file as input
wf.inputs.inputnode.seg_file = seg_file
wf.base_dir = str(output_dir)

# Run the workflow
wf.run()
