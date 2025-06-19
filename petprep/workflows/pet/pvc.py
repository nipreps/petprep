from __future__ import annotations

import json
from pathlib import Path
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.petpvc import PETPVC
from nipype.interfaces.freesurfer.petsurfer import GTMPVC
from nipype.interfaces.fsl import Split, Merge
import nibabel as nb
from nipype.interfaces.freesurfer import ApplyVolTransform

from petprep.interfaces.pvc import CSVtoNifti, StackTissueProbabilityMaps, Binarise4DSegmentation

def load_pvc_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

def init_pet_pvc_wf(
    *,
    tool: str = 'PETPVC',
    method: str = 'GTM',
    pvc_params: dict | None = None,
    config_path: Path,
    name: str = 'pet_pvc_wf',
) -> pe.Workflow:
    config = load_pvc_config(config_path)

    tool_lower = tool.lower()
    method_key = method.upper() if tool_lower == 'petpvc' else method.lower()

    if method_key not in config.get(tool_lower, {}):
        raise ValueError(f"Method '{method}' is not valid for tool '{tool}'.")

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'anat_seg', 'anat_file', 'reg_file', 't1w_tpms']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_pvc_file']),
        name='outputnode'
    )

    pvc_params = pvc_params or {}
    method_config = config[tool_lower][method_key].copy()
    method_config.update(pvc_params)

    if tool_lower == 'petpvc':
        # Handling 4D PETPVC processing
        split_frames = pe.Node(niu.Split(dimension='t'), name='split_frames')
        merge_frames = pe.Node(niu.Merge(dimension='t'), name='merge_frames')

        resample_pet_to_anat = pe.MapNode(
            ApplyVolTransform(interp='nearest', reg_header=True),
            iterfield=['source_file'],
            name='resample_pet'
        )

        pvc_node = pe.MapNode(
            PETPVC(pvc=method_config.pop('pvc'), **method_config),
            iterfield=['in_file'],
            name=f'{tool_lower}_{method_key.lower()}_pvc_node',
        )

        resample_pet_to_petref = pe.MapNode(
            ApplyVolTransform(interp='nearest', reg_header=True),
            iterfield=['source_file'],
            name='resample_pet_to_petref'
        )

        workflow.connect([
        (inputnode, split_frames, [('pet_file', 'in_file')]),
        (split_frames, resample_pet_to_anat, [('out_files', 'source_file')]),
        (inputnode, resample_pet_to_anat, [('anat_seg', 'target_file')]),
        (resample_pet_to_anat, pvc_node, [('transformed_file', 'in_file')]),
    ])

        if method_key == 'MG':
            stack_node = pe.Node(StackTissueProbabilityMaps(), name='stack_probmaps')
            workflow.connect([
                (inputnode, stack_node, [('t1w_tpms', 't1w_tpms')]),
                (stack_node, pvc_node, [('out_file', 'mask_file')]),
                (pvc_node, merge_frames, [('out_file', 'in_files')]),
                (merge_frames, outputnode, [('merged_file', 'pet_pvc_file')]),
            ])

        elif method_key == 'GTM':
            pvc_node.inputs.out_file = 'gtm_output.csv'

            csv_to_nifti_node = pe.MapNode(
                CSVtoNifti(),
                iterfield=['csv_file'],
                name='csv_to_nifti_node'
            )

            workflow.connect([
                (inputnode, pvc_node, [('anat_seg', 'mask_file')]),
                (pvc_node, csv_to_nifti_node, [('out_file', 'csv_file')]),
                (inputnode, csv_to_nifti_node, [('anat_seg', 'reference_nifti')]),
                (csv_to_nifti_node, merge_frames, [('out_file', 'in_files')]),
                (merge_frames, outputnode, [('merged_file', 'pet_pvc_file')]),
            ])

        else:
            workflow.connect([
                (inputnode, pvc_node, [('anat_seg', 'mask_file')]),
                (pvc_node, merge_frames, [('out_file', 'in_files')]),
                (merge_frames, outputnode, [('merged_file', 'pet_pvc_file')]),
            ])

        workflow.connect([
            (merge_frames, resample_pet_to_petref, [('merged_file', 'source_file')]),
            (inputnode, resample_pet_to_petref, [('pet_file', 'target_file')]),
            (resample_pet_to_petref, outputnode, [('transformed_file', 'pet_pvc_file')]),
        ])

    elif tool_lower == 'petsurfer':
        # PETSurfer directly handles 4D data (no splitting needed)
        pvc_node = pe.Node(
            GTMPVC(**method_config),
            name=f'{tool_lower}_{method_key.lower()}_pvc_node',
        )

        workflow.connect([
            (inputnode, pvc_node, [
                ('pet_file', 'in_file'),
                ('anat_seg', 'segmentation'),
                ('reg_file', 'reg_file')
            ]),
            (pvc_node, outputnode, [('gtm_file', 'pet_pvc_file')]),
        ])

    else:
        raise ValueError(f"Unsupported PVC tool: {tool}")

    return workflow
