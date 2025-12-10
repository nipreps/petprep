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
Registration workflows
++++++++++++++++++++++

.. autofunction:: init_pet_reg_wf

"""

import typing as ty

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe

from petprep import config

AffineDOF = ty.Literal[6, 9, 12]


def _get_first(in_list):
    """Extract first element from a list (for ANTs transform output)."""
    return in_list[0]


def _select_best_transform(xfm_ants, xfm_fs, inv_ants, inv_fs, score_ants, score_fs):
    """Pick the transform with the highest similarity score."""

    # Default to FreeSurfer branch if scores tie
    if score_ants > score_fs:
        return xfm_ants, inv_ants, 'ants', score_ants
    return xfm_fs, inv_fs, 'freesurfer', score_fs


def init_pet_reg_wf(
    *,
    pet2anat_dof: AffineDOF,
    mem_gb: float,
    omp_nthreads: int,
    pet2anat_method: str = 'mri_coreg',
    name: str = 'pet_reg_wf',
    sloppy: bool = False,
):
    """
    Build a workflow to run same-subject, PET-to-anat image-registration.

    Calculates the registration between a reference PET image and anat-space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.pet.registration import init_pet_reg_wf
            wf = init_pet_reg_wf(
                mem_gb=3,
                omp_nthreads=1,
                pet2anat_dof=6,
            )

    Parameters
    ----------
    pet2anat_dof : 6, 9 or 12
        Degrees-of-freedom for PET-anatomical registration
    mem_gb : :obj:`float`
        Size of PET file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    pet2anat_method : :obj:`str`
        Method for PET-to-anatomical registration. Options are 'mri_coreg'
        (default FreeSurfer co-registration), 'robust' (uses FreeSurfer
        mri_robust_register with NMI, 6 DoF only), 'ants' (uses ANTs rigid
        registration, 6 DoF only), or 'auto' (runs FreeSurfer and ANTs in
        parallel, selecting the best-performing transform).
    name : :obj:`str`
        Name of workflow (default: ``pet_reg_wf``)

    Inputs
    ------
    ref_pet_brain
        Reference image to which PET series is aligned
        If ``fieldwarp == True``, ``ref_pet_brain`` should be unwarped
    anat_preproc
        Preprocessed anatomical image
    anat_mask
        Brainmask for anatomical image

    Outputs
    -------
    itk_pet_to_anat
        Affine transform from ``ref_pet_brain`` to anatomical space (ITK format)
    itk_anat_to_pet
        Affine transform from anatomical space to PET space (ITK format)
    registration_winner
        Name of the registration backend selected when ``pet2anat_method='auto'``

    """
    from nipype.interfaces.ants import MeasureImageSimilarity, Registration
    from nipype.interfaces.freesurfer import MRIConvert, MRICoreg, RobustRegister
    from nipype.interfaces.fsl import RobustFOV
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['ref_pet_brain', 'anat_preproc', 'anat_mask']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['itk_pet_to_t1', 'itk_t1_to_pet', 'registration_winner', 'registration_score']
        ),
        name='outputnode',
    )
    outputnode.inputs.registration_winner = None
    outputnode.inputs.registration_score = None

    mask_brain = pe.Node(ApplyMask(), name='mask_brain')
    crop_anat_mask = pe.Node(MRIConvert(out_type='niigz'), name='crop_anat_mask')
    robust_fov = pe.Node(RobustFOV(output_type='NIFTI_GZ'), name='robust_fov')

    if pet2anat_method == 'auto':
        ants_coreg = pe.Node(
            Registration(
                dimension=3,
                float=True,
                output_transform_prefix='pet2anat_',
                output_warped_image='pet2anat_Warped.nii.gz',
                transforms=['Rigid'],
                transform_parameters=[(0.1,)],
                metric=['MI'],
                metric_weight=[1],
                radius_or_number_of_bins=[32],
                sampling_strategy=['Regular'],
                sampling_percentage=[0.25],
                number_of_iterations=[[1000, 500, 250]],
                convergence_threshold=[1e-6],
                convergence_window_size=[10],
                shrink_factors=[[4, 2, 1]],
                smoothing_sigmas=[[2, 1, 0]],
                sigma_units=['vox'],
                use_histogram_matching=False,
                initial_moving_transform_com=1,
                winsorize_lower_quantile=0.001,
                winsorize_upper_quantile=0.999,
            ),
            name='ants_registration',
            n_procs=omp_nthreads,
            mem_gb=mem_gb * 2,
        )
        fs_coreg = pe.Node(
            MRICoreg(dof=pet2anat_dof, sep=[4], ftol=0.0001, linmintol=0.01),
            name='mri_coreg',
            n_procs=omp_nthreads,
            mem_gb=5,
        )

        ants_convert = pe.Node(ConcatenateXFMs(inverse=True), name='convert_xfm_ants')
        fs_convert = pe.Node(ConcatenateXFMs(inverse=True), name='convert_xfm_fs')

        ants_warp = pe.Node(ApplyTransforms(float=True), name='warp_pet_ants')
        fs_warp = pe.Node(ApplyTransforms(float=True), name='warp_pet_fs')

        ants_score = pe.Node(
            MeasureImageSimilarity(
                metric='Mattes',
                dimension=3,
                radius_or_number_of_bins=32,
                sampling_strategy='Regular',
                sampling_percentage=0.25,
            ),
            name='score_ants',
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        fs_score = pe.Node(
            MeasureImageSimilarity(
                metric='Mattes',
                dimension=3,
                radius_or_number_of_bins=32,
                sampling_strategy='Regular',
                sampling_percentage=0.25,
            ),
            name='score_fs',
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )

        select_best = pe.Node(
            niu.Function(
                function=_select_best_transform,
                input_names=['xfm_ants', 'xfm_fs', 'inv_ants', 'inv_fs', 'score_ants', 'score_fs'],
                output_names=['best_xfm', 'best_inv_xfm', 'winner', 'best_score'],
            ),
            name='select_best',
        )

        workflow.connect(
            [
                (inputnode, robust_fov, [('anat_preproc', 'in_file')]),
                (inputnode, crop_anat_mask, [('anat_mask', 'in_file')]),
                (robust_fov, crop_anat_mask, [('out_roi', 'reslice_like')]),
                (robust_fov, mask_brain, [('out_roi', 'in_file')]),
                (crop_anat_mask, mask_brain, [('out_file', 'in_mask')]),
                # ANTs branch
                (inputnode, ants_coreg, [('ref_pet_brain', 'moving_image')]),
                (robust_fov, ants_coreg, [('out_roi', 'fixed_image')]),
                (crop_anat_mask, ants_coreg, [('out_file', 'fixed_image_masks')]),
                (ants_coreg, ants_convert, [(('forward_transforms', _get_first), 'in_xfms')]),
                (inputnode, ants_warp, [('ref_pet_brain', 'input_image')]),
                (robust_fov, ants_warp, [('out_roi', 'reference_image')]),
                (ants_convert, ants_warp, [('out_xfm', 'transforms')]),
                (ants_warp, ants_score, [('output_image', 'moving_image')]),
                (mask_brain, ants_score, [('out_file', 'fixed_image')]),
                (crop_anat_mask, ants_score, [('out_file', 'fixed_image_mask')]),
                (crop_anat_mask, ants_score, [('out_file', 'moving_image_mask')]),
                # FreeSurfer branch
                (inputnode, fs_coreg, [('ref_pet_brain', 'source_file')]),
                (mask_brain, fs_coreg, [('out_file', 'reference_file')]),
                (fs_coreg, fs_convert, [('out_lta_file', 'in_xfms')]),
                (inputnode, fs_warp, [('ref_pet_brain', 'input_image')]),
                (mask_brain, fs_warp, [('out_file', 'reference_image')]),
                (fs_convert, fs_warp, [('out_xfm', 'transforms')]),
                (fs_warp, fs_score, [('output_image', 'moving_image')]),
                (mask_brain, fs_score, [('out_file', 'fixed_image')]),
                (crop_anat_mask, fs_score, [('out_file', 'fixed_image_mask')]),
                (crop_anat_mask, fs_score, [('out_file', 'moving_image_mask')]),
                # Selection
                (ants_convert, select_best, [('out_xfm', 'xfm_ants'), ('out_inv', 'inv_ants')]),
                (fs_convert, select_best, [('out_xfm', 'xfm_fs'), ('out_inv', 'inv_fs')]),
                (ants_score, select_best, [('similarity', 'score_ants')]),
                (fs_score, select_best, [('similarity', 'score_fs')]),
                (select_best, outputnode, [
                    ('best_xfm', 'itk_pet_to_t1'),
                    ('best_inv_xfm', 'itk_t1_to_pet'),
                    ('winner', 'registration_winner'),
                    ('best_score', 'registration_score'),
                ]),
            ]
        )  # fmt:skip

        return workflow

    if pet2anat_method == 'ants':
        coreg = pe.Node(
            Registration(
                dimension=3,
                float=True,
                output_transform_prefix='pet2anat_',
                output_warped_image='pet2anat_Warped.nii.gz',
                transforms=['Rigid'],
                transform_parameters=[(0.1,)],
                metric=['MI'],
                metric_weight=[1],
                radius_or_number_of_bins=[32],
                sampling_strategy=['Regular'],
                sampling_percentage=[0.25],
                number_of_iterations=[[1000, 500, 250]],
                convergence_threshold=[1e-6],
                convergence_window_size=[10],
                shrink_factors=[[4, 2, 1]],
                smoothing_sigmas=[[2, 1, 0]],
                sigma_units=['vox'],
                use_histogram_matching=False,
                initial_moving_transform_com=1,
                winsorize_lower_quantile=0.001,
                winsorize_upper_quantile=0.999,
            ),
            name='ants_registration',
            n_procs=omp_nthreads,
            mem_gb=mem_gb * 2,
        )
        coreg_target = 'fixed_image'
        coreg_mask = 'fixed_image_masks'
        coreg_moving = 'moving_image'
        coreg_output = 'forward_transforms'
        coreg_output_is_list = True

    elif pet2anat_method == 'robust':
        coreg = pe.Node(
            RobustRegister(
                auto_sens=False,
                est_int_scale=False,
                init_orient=True,
                args='--cost NMI',
                max_iterations=10,
                high_iterations=20,
                iteration_thresh=0.01,
            ),
            name='mri_robust_register',
            n_procs=omp_nthreads,
            mem_gb=5,
        )
        coreg_target = 'target_file'
        coreg_moving = 'source_file'
        coreg_output = 'out_reg_file'
        coreg_output_is_list = False

    else:  # mri_coreg (default)
        coreg = pe.Node(
            MRICoreg(dof=pet2anat_dof, sep=[4], ftol=0.0001, linmintol=0.01),
            name='mri_coreg',
            n_procs=omp_nthreads,
            mem_gb=5,
        )
        coreg_target = 'reference_file'
        coreg_moving = 'source_file'
        coreg_output = 'out_lta_file'
        coreg_output_is_list = False

    convert_xfm = pe.Node(ConcatenateXFMs(inverse=True), name='convert_xfm')
    warp_for_score = pe.Node(ApplyTransforms(float=True), name='warp_for_score')
    similarity = pe.Node(
        MeasureImageSimilarity(
            metric='Mattes',
            dimension=3,
            radius_or_number_of_bins=32,
            sampling_strategy='Regular',
            sampling_percentage=0.25,
        ),
        name='score_registration',
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    # Build connections dynamically based on output type
    if coreg_output_is_list:
        # ANTs outputs a list of transforms; take the first (and only) one
        # ANTs gets unmasked T1W + separate mask (not pre-masked image)
        connections = [
            (robust_fov, mask_brain, [('out_roi', 'in_file')]),
            (crop_anat_mask, mask_brain, [('out_file', 'in_mask')]),
            (inputnode, coreg, [('ref_pet_brain', coreg_moving)]),
            (
                robust_fov,
                coreg,
                [
                    ('out_roi', coreg_target),
                ],
            ),
            (crop_anat_mask, coreg, [('out_file', coreg_mask)]),
            (coreg, convert_xfm, [((coreg_output, _get_first), 'in_xfms')]),
            (
                convert_xfm,
                outputnode,
                [
                    ('out_xfm', 'itk_pet_to_t1'),
                    ('out_inv', 'itk_t1_to_pet'),
                ],
            ),
            (inputnode, warp_for_score, [('ref_pet_brain', 'input_image')]),
            (robust_fov, warp_for_score, [('out_roi', 'reference_image')]),
            (convert_xfm, warp_for_score, [('out_xfm', 'transforms')]),
            (warp_for_score, similarity, [('output_image', 'moving_image')]),
            (mask_brain, similarity, [('out_file', 'fixed_image')]),
            (crop_anat_mask, similarity, [('out_file', 'fixed_image_mask')]),
            (crop_anat_mask, similarity, [('out_file', 'moving_image_mask')]),
            (similarity, outputnode, [('similarity', 'registration_score')]),
        ]
    else:
        # mri_coreg and mri_robust_register output single transform file
        connections = [
            (robust_fov, mask_brain, [('out_roi', 'in_file')]),
            (crop_anat_mask, mask_brain, [('out_file', 'in_mask')]),
            (inputnode, coreg, [('ref_pet_brain', coreg_moving)]),
            (mask_brain, coreg, [('out_file', coreg_target)]),
            (coreg, convert_xfm, [(coreg_output, 'in_xfms')]),
            (
                convert_xfm,
                outputnode,
                [
                    ('out_xfm', 'itk_pet_to_t1'),
                    ('out_inv', 'itk_t1_to_pet'),
                ],
            ),
            (inputnode, warp_for_score, [('ref_pet_brain', 'input_image')]),
            (robust_fov, warp_for_score, [('out_roi', 'reference_image')]),
            (convert_xfm, warp_for_score, [('out_xfm', 'transforms')]),
            (warp_for_score, similarity, [('output_image', 'moving_image')]),
            (mask_brain, similarity, [('out_file', 'fixed_image')]),
            (crop_anat_mask, similarity, [('out_file', 'fixed_image_mask')]),
            (crop_anat_mask, similarity, [('out_file', 'moving_image_mask')]),
            (similarity, outputnode, [('similarity', 'registration_score')]),
        ]

    workflow.connect(
        [
            (inputnode, robust_fov, [('anat_preproc', 'in_file')]),
            (inputnode, crop_anat_mask, [('anat_mask', 'in_file')]),
            (robust_fov, crop_anat_mask, [('out_roi', 'reslice_like')]),
        ]
        + connections
    )  # fmt:skip

    return workflow
