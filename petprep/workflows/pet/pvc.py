from __future__ import annotations

import json
from pathlib import Path
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.petpvc import PETPVC
from nipype.interfaces.fsl import Split, Merge
import nibabel as nb
from nipype.interfaces.freesurfer import ApplyVolTransform, Tkregister2, MRICoreg

from petprep.interfaces.pvc import CSVtoNifti, StackTissueProbabilityMaps, Binarise4DSegmentation, GTMPVC, GTMStatsTo4DNifti


def load_pvc_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


# Add a function to dynamically construct the path
def construct_gtmseg_path(subjects_dir, subject_id):
    from pathlib import Path
    return str(Path(subjects_dir) / subject_id / 'mri' / 'gtmseg.mgz')


def construct_nu_path(subjects_dir, subject_id):
    from pathlib import Path
    return str(Path(subjects_dir) / subject_id / 'mri' / 'nu.mgz')


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
    method_key = method.upper()

    if method_key not in config.get(tool_lower, {}):
        raise ValueError(f"Method '{method}' is not valid for tool '{tool}'.")

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'segmentation', 't1w_tpms', 'petref', 'subjects_dir', 'subject_id']),
        name='inputnode'
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_pvc_file', 'pet_pvc_mask']),
        name='outputnode'
    )

    pvc_params = pvc_params or {}
    method_config = config[tool_lower][method_key].copy()
    method_config.update(pvc_params)

    if tool_lower == 'petpvc':
        # Handling 4D PETPVC processing
        split_frames = pe.Node(Split(dimension='t'), name='split_frames')
        merge_frames = pe.Node(Merge(dimension='t'), name='merge_frames')

        resample_pet_to_anat = pe.MapNode(
            ApplyVolTransform(interp='nearest', reg_header=True),
            iterfield=['source_file'],
            name='resample_pet_to_anat'
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
            (inputnode, resample_pet_to_anat, [('segmentation', 'target_file')]),
            (resample_pet_to_anat, pvc_node, [('transformed_file', 'in_file')]),
        ])

        if method_key == 'MG':
            stack_node = pe.Node(StackTissueProbabilityMaps(), name='stack_probmaps')
            workflow.connect([
                (inputnode, stack_node, [('t1w_tpms', 't1w_tpms')]),
                (stack_node, pvc_node, [('out_file', 'mask_file')]),
                (pvc_node, merge_frames, [('out_file', 'in_files')]),
            ])

        else:
            binarise_segmentation = pe.Node(Binarise4DSegmentation(), name='binarise_segmentation')
            workflow.connect([
                (inputnode, binarise_segmentation, [('segmentation', 'dseg_file')]),
                (binarise_segmentation, pvc_node, [('out_file', 'mask_file')]),
            ])

            if method_key == 'GTM':
                pvc_node.inputs.out_file = 'gtm_output.csv'

                csv_to_nifti_node = pe.MapNode(
                    CSVtoNifti(),
                    iterfield=['csv_file'],
                    name='csv_to_nifti_node'
                )

                workflow.connect([
                    (pvc_node, csv_to_nifti_node, [('out_file', 'csv_file')]),
                    (binarise_segmentation, csv_to_nifti_node, [('label_list', 'label_list')]),
                    (inputnode, csv_to_nifti_node, [('segmentation', 'reference_nifti')]),
                    (csv_to_nifti_node, merge_frames, [('out_file', 'in_files')]),
                ])

            else:
                workflow.connect([
                    (pvc_node, merge_frames, [('out_file', 'in_files')]),
                ])

        workflow.connect([
            (merge_frames, resample_pet_to_petref, [('merged_file', 'source_file')]),
            (inputnode, resample_pet_to_petref, [('pet_file', 'target_file')]),
            (resample_pet_to_petref, outputnode, [('transformed_file', 'pet_pvc_file')]),
        ])

    elif tool_lower == 'petsurfer' and method_key == 'GTM' or method_key == 'MG' or method_key == 'RBV':
        # PETSurfer directly handles 4D data (no splitting needed)
        tkregister_node = pe.Node(
            Tkregister2(
                reg_file='identity.dat',
                reg_header=True,
                lta_out='identity_vox.lta',
            ),
            name='tkregister_identity'
        )

        gtmseg_path_node = pe.Node(
            niu.Function(
                input_names=['subjects_dir', 'subject_id'],
                output_names=['gtmseg_path'],
                function=construct_gtmseg_path
            ),
            name='gtmseg_path'
        )

        nu_path_node = pe.Node(
            niu.Function(
                input_names=['subjects_dir', 'subject_id'],
                output_names=['nu_path'],
                function=construct_nu_path
            ),
            name='nu_path'
        )

        if 'auto_mask' in method_config:
            method_config['auto_mask'] = tuple(method_config['auto_mask'])

        if 'mg' in method_config:
            method_config['mg'] = tuple(method_config['mg'])

        pvc_node = pe.Node(
            GTMPVC(**method_config),
            name=f'{tool_lower}_{method_key.lower()}_pvc_node',
        )

        workflow.connect([
            (inputnode, nu_path_node, [
                ('subjects_dir', 'subjects_dir'),
                ('subject_id', 'subject_id'),
            ]),
            (inputnode, tkregister_node, [('petref', 'moving_image')]),
            (nu_path_node, tkregister_node, [('nu_path', 'target_image')]),
            (inputnode, tkregister_node, [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')]),
            (tkregister_node, pvc_node, [('lta_file', 'reg_file')]),
            (inputnode, gtmseg_path_node, [
                ('subjects_dir', 'subjects_dir'),
                ('subject_id', 'subject_id'),
            ]),
            (inputnode, pvc_node, [
                ('pet_file', 'in_file'),
                ('subjects_dir', 'subjects_dir'),
            ]),
            (gtmseg_path_node, pvc_node, [('gtmseg_path', 'segmentation')]),
        ])

        # Conditional output based on method
        if method_key == 'GTM':

            gtm_stats_node = pe.Node(
                GTMStatsTo4DNifti(),
                name='gtm_stats_to_4d_nifti'
            )

            workflow.connect([
                (pvc_node, gtm_stats_node, [('gtm_file', 'gtm_file'),
                                            ('gtm_stats', 'gtm_stats')]),
                (inputnode, gtm_stats_node, [('segmentation', 'segmentation')]),
            ])

            workflow.connect([
                (pvc_node, outputnode, [('tissue_fraction', 'pet_pvc_mask')]),
                (gtm_stats_node, outputnode, [('out_file', 'pet_pvc_file')]),
            ])

        elif method_key == 'MG':
            workflow.connect([(pvc_node, outputnode, [('mg', 'pet_pvc_file')])])

        elif method_key == 'RBV':
            workflow.connect([(pvc_node, outputnode, [('rbv', 'pet_pvc_file')])])

        #workflow.connect([(pvc_node, outputnode, [('tissue_fraction', 'pet_pvc_mask')])])

    else:
        raise ValueError(f"Unsupported method PVC ({method}) for PVC tool: {tool}")

    return workflow
