# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""
Partial Volume Correction
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_pet_pvc_wf

"""

from __future__ import annotations

import json
import re
from pathlib import Path

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.freesurfer import ApplyVolTransform, Tkregister2
from nipype.interfaces.fsl import MeanImage, Merge, Split
from nipype.interfaces.petpvc import PETPVC

from petprep.interfaces.pvc import (
    GTMPVC,
    Binarise4DSegmentation,
    ClipValues,
    CSVtoNifti,
    GTMStatsTo4DNifti,
    StackTissueProbabilityMaps,
    get_opt_fwhm,
)


def load_pvc_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


# Add a function to dynamically construct the path
def construct_gtmseg_path(subjects_dir, subject_id, seg_ready=None):
    from pathlib import Path

    return str(Path(subjects_dir) / subject_id / 'mri' / 'gtmseg.mgz')


def construct_nu_path(subjects_dir, subject_id):
    from pathlib import Path

    return str(Path(subjects_dir) / subject_id / 'mri' / 'nu.mgz')


def sanitize_name(name: str) -> str:
    """Ensure names are Nipype-compatible by replacing invalid characters with underscores."""
    return re.sub(r'\W+', '_', name.lower())


def init_pet_pvc_wf(
    *,
    tool: str = 'PETPVC',
    method: str = 'GTM',
    pvc_params: dict | None = None,
    config_path: Path,
    name: str = 'pet_pvc_wf',
) -> pe.Workflow:
    r"""
    Build a workflow to perform partial volume correction.

    This workflow applies :abbr:`PVC (partial volume correction)` to the input
    :abbr:`PET (positron emission tomography)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.pet import init_pet_pvc_wf
            wf = init_pet_pvc_wf(
                tool='PETPVC',
            )
    """
    config = load_pvc_config(config_path)

    tool_lower = tool.lower()
    method_key = method.upper()
    safe_method = sanitize_name(method)

    if method_key not in config.get(tool_lower, {}):
        raise ValueError(f"Method '{method}' is not valid for tool '{tool}'.")

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['pet_file', 'segmentation', 't1w_tpms', 'petref', 'subjects_dir', 'subject_id']
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_pvc_file',
                'pet_pvc_mask',
                'fwhm_x',
                'fwhm_y',
                'fwhm_z',
            ]
        ),
        name='outputnode',
    )

    pvc_params = pvc_params or {}
    method_config = config[tool_lower][method_key].copy()
    method_config.update(pvc_params)

    const_psf = pe.Node(
        niu.IdentityInterface(fields=['fwhm_x', 'fwhm_y', 'fwhm_z']),
        name='const_psf',
    )
    if 'psf' in method_config:
        const_psf.inputs.fwhm_x = const_psf.inputs.fwhm_y = const_psf.inputs.fwhm_z = (
            method_config['psf']
        )
    else:
        const_psf.inputs.fwhm_x = method_config.get('fwhm_x')
        const_psf.inputs.fwhm_y = method_config.get('fwhm_y')
        const_psf.inputs.fwhm_z = method_config.get('fwhm_z')

    resample_pet_to_petref = pe.Node(
        ApplyVolTransform(interp='nearest', reg_header=True),
        iterfield=['source_file'],
        name='resample_pet_to_petref',
    )

    if tool_lower == 'petpvc':
        # Handling 4D PETPVC processing
        split_frames = pe.Node(Split(dimension='t'), name='split_frames')
        merge_frames = pe.Node(Merge(dimension='t'), name='merge_frames')

        resample_pet_to_anat = pe.MapNode(
            ApplyVolTransform(interp='nearest', reg_header=True),
            iterfield=['source_file'],
            name='resample_pet_to_anat',
        )

        clip_values = pe.MapNode(ClipValues(), iterfield=['in_file'], name='clip_values')

        pvc_node = pe.MapNode(
            PETPVC(pvc=method_config.pop('pvc'), **method_config),
            iterfield=['in_file'],
            name=f'{tool_lower}_{safe_method}_pvc_node',
        )

        workflow.connect(
            [
                (inputnode, split_frames, [('pet_file', 'in_file')]),
                (split_frames, resample_pet_to_anat, [('out_files', 'source_file')]),
                (inputnode, resample_pet_to_anat, [('segmentation', 'target_file')]),
                (resample_pet_to_anat, clip_values, [('transformed_file', 'in_file')]),
                (clip_values, pvc_node, [('out_file', 'in_file')]),
            ]
        )

        if method_key == 'MG':
            stack_node = pe.Node(StackTissueProbabilityMaps(), name='stack_probmaps')
            workflow.connect(
                [
                    (inputnode, stack_node, [('t1w_tpms', 't1w_tpms')]),
                    (stack_node, pvc_node, [('out_file', 'mask_file')]),
                    (pvc_node, merge_frames, [('out_file', 'in_files')]),
                ]
            )

        else:
            binarise_segmentation = pe.Node(Binarise4DSegmentation(), name='binarise_segmentation')
            workflow.connect(
                [
                    (inputnode, binarise_segmentation, [('segmentation', 'dseg_file')]),
                    (binarise_segmentation, pvc_node, [('out_file', 'mask_file')]),
                ]
            )

            if method_key == 'GTM':
                pvc_node.inputs.out_file = 'gtm_output.csv'

                csv_to_nifti_node = pe.MapNode(
                    CSVtoNifti(), iterfield=['csv_file'], name='csv_to_nifti_node'
                )

                workflow.connect(
                    [
                        (pvc_node, csv_to_nifti_node, [('out_file', 'csv_file')]),
                        (binarise_segmentation, csv_to_nifti_node, [('label_list', 'label_list')]),
                        (inputnode, csv_to_nifti_node, [('segmentation', 'reference_nifti')]),
                        (csv_to_nifti_node, merge_frames, [('out_file', 'in_files')]),
                    ]
                )

            else:
                workflow.connect(
                    [
                        (pvc_node, merge_frames, [('out_file', 'in_files')]),
                    ]
                )

        workflow.connect(
            [
                (merge_frames, resample_pet_to_petref, [('merged_file', 'source_file')]),
                (inputnode, resample_pet_to_petref, [('pet_file', 'target_file')]),
                (resample_pet_to_petref, outputnode, [('transformed_file', 'pet_pvc_file')]),
            ]
        )

    elif tool_lower == 'petsurfer' and method_key in ('GTM', 'MG', 'RBV', 'AGTM'):
        # PETSurfer directly handles 4D data (no splitting needed)
        tkregister_node = pe.Node(
            Tkregister2(
                reg_file='identity.dat',
                reg_header=True,
                lta_out='identity_vox.lta',
            ),
            name='tkregister_identity',
        )

        gtmseg_path_node = pe.Node(
            niu.Function(
                input_names=['subjects_dir', 'subject_id', 'seg_ready'],
                output_names=['gtmseg_path'],
                function=construct_gtmseg_path,
            ),
            name='gtmseg_path',
        )

        nu_path_node = pe.Node(
            niu.Function(
                input_names=['subjects_dir', 'subject_id'],
                output_names=['nu_path'],
                function=construct_nu_path,
            ),
            name='nu_path',
        )

        if 'auto_mask' in method_config:
            method_config['auto_mask'] = tuple(method_config['auto_mask'])

        if 'mg' in method_config:
            method_config['mg'] = tuple(method_config['mg'])

        if 'opt_tol' in method_config:
            method_config['opt_tol'] = tuple(method_config['opt_tol'])

        if method_key == 'AGTM':
            mean_pet = pe.Node(MeanImage(dimension='T'), name='mean_pet')

            est_node = pe.Node(
                GTMPVC(**method_config),
                name=f'{tool_lower}_{method_key.lower()}_estimate_psf',
            )

            apply_config = method_config.copy()
            apply_config.pop('optimization_schema', None)
            apply_config.pop('opt_brain', None)
            apply_config.pop('opt_seg_merge', None)
            apply_config.pop('opt_tol', None)
            apply_config.pop('psf', None)

            pvc_node = pe.Node(
                GTMPVC(**apply_config),
                name=f'{tool_lower}_{method_key.lower()}_pvc_node',
            )

            get_fwhm = pe.Node(
                niu.Function(
                    input_names=['opt_params'],
                    output_names=['psf_col', 'psf_row', 'psf_slice'],
                    function=get_opt_fwhm,
                ),
                name='get_opt_fwhm',
            )
        else:
            pvc_node = pe.Node(
                GTMPVC(**method_config),
                name=f'{tool_lower}_{method_key.lower()}_pvc_node',
            )

        workflow.connect(
            [
                (
                    inputnode,
                    nu_path_node,
                    [
                        ('subjects_dir', 'subjects_dir'),
                        ('subject_id', 'subject_id'),
                    ],
                ),
                (inputnode, tkregister_node, [('pet_file', 'moving_image')]),
                (nu_path_node, tkregister_node, [('nu_path', 'target_image')]),
                (
                    inputnode,
                    tkregister_node,
                    [('subjects_dir', 'subjects_dir'), ('subject_id', 'subject_id')],
                ),
            ]
        )

        if method_key == 'AGTM':
            workflow.connect(
                [
                    (tkregister_node, est_node, [('lta_file', 'reg_file')]),
                    (
                        inputnode,
                        gtmseg_path_node,
                        [
                            ('subjects_dir', 'subjects_dir'),
                            ('subject_id', 'subject_id'),
                            ('segmentation', 'seg_ready'),
                        ],
                    ),
                    (inputnode, mean_pet, [('pet_file', 'in_file')]),
                    (mean_pet, est_node, [('out_file', 'in_file')]),
                    (inputnode, est_node, [('subjects_dir', 'subjects_dir')]),
                    (gtmseg_path_node, est_node, [('gtmseg_path', 'segmentation')]),
                    (est_node, get_fwhm, [('opt_params', 'opt_params')]),
                    (tkregister_node, pvc_node, [('lta_file', 'reg_file')]),
                    (
                        inputnode,
                        pvc_node,
                        [
                            ('pet_file', 'in_file'),
                            ('subjects_dir', 'subjects_dir'),
                        ],
                    ),
                    (gtmseg_path_node, pvc_node, [('gtmseg_path', 'segmentation')]),
                    (
                        get_fwhm,
                        pvc_node,
                        [
                            ('psf_col', 'psf_col'),
                            ('psf_row', 'psf_row'),
                            ('psf_slice', 'psf_slice'),
                        ],
                    ),
                    (
                        get_fwhm,
                        outputnode,
                        [
                            ('psf_col', 'fwhm_x'),
                            ('psf_row', 'fwhm_y'),
                            ('psf_slice', 'fwhm_z'),
                        ],
                    ),
                ]
            )
        else:
            workflow.connect(
                [
                    (tkregister_node, pvc_node, [('lta_file', 'reg_file')]),
                    (
                        inputnode,
                        gtmseg_path_node,
                        [
                            ('subjects_dir', 'subjects_dir'),
                            ('subject_id', 'subject_id'),
                            ('segmentation', 'seg_ready'),
                        ],
                    ),
                    (
                        inputnode,
                        pvc_node,
                        [
                            ('pet_file', 'in_file'),
                            ('subjects_dir', 'subjects_dir'),
                        ],
                    ),
                    (gtmseg_path_node, pvc_node, [('gtmseg_path', 'segmentation')]),
                ]
            )

            workflow.connect(
                [
                    (
                        const_psf,
                        outputnode,
                        [
                            ('fwhm_x', 'fwhm_x'),
                            ('fwhm_y', 'fwhm_y'),
                            ('fwhm_z', 'fwhm_z'),
                        ],
                    )
                ]
            )

        # Conditional output based on method
        if method_key in ('GTM', 'AGTM'):
            gtm_stats_node = pe.Node(GTMStatsTo4DNifti(), name='gtm_stats_to_4d_nifti')

            workflow.connect(
                [
                    (
                        pvc_node,
                        gtm_stats_node,
                        [('gtm_file', 'gtm_file'), ('gtm_stats', 'gtm_stats')],
                    ),
                    (inputnode, gtm_stats_node, [('segmentation', 'segmentation')]),
                ]
            )

            workflow.connect([(gtm_stats_node, outputnode, [('out_file', 'pet_pvc_file')])])

        elif method_key == 'MG':
            workflow.connect([(pvc_node, outputnode, [('mg', 'pet_pvc_file')])])

        elif method_key == 'RBV':
            workflow.connect([(pvc_node, outputnode, [('rbv', 'pet_pvc_file')])])

        workflow.connect([(pvc_node, outputnode, [('tissue_fraction', 'pet_pvc_mask')])])

    else:
        raise ValueError(f'Unsupported method PVC ({method}) for PVC tool: {tool}')

    return workflow
