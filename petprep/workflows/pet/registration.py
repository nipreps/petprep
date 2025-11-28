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

AffineDOF = ty.Literal[6, 9, 12]


def init_pet_reg_wf(
    *,
    pet2anat_dof: AffineDOF,
    mem_gb: float,
    omp_nthreads: int,
    use_robust_register: bool = False,
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
    use_robust_register : :obj:`bool`
        Run an ANTs mutual-information rigid registration for PET-to-
        anatomical alignment. Only rigid-body (6 dof) alignment is supported
        in this mode.
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

    """
    from nipype.interfaces.ants import Registration
    from nipype.interfaces.freesurfer import MRICoreg
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow
    from niworkflows.interfaces.nibabel import ApplyMask
    from niworkflows.interfaces.nitransforms import ConcatenateXFMs

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['ref_pet_brain', 'anat_preproc', 'anat_mask']),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['itk_pet_to_t1', 'itk_t1_to_pet']),
        name='outputnode',
    )

    mask_brain = pe.Node(ApplyMask(), name='mask_brain')
    if use_robust_register:
        coreg = pe.Node(
            Registration(
                dimension=3,
                float=False,
                output_transform_prefix='pet2anat_',
                transforms=['Rigid'],
                transform_parameters=[(0.1,)],
                number_of_iterations=[[1000, 500, 250]],
                metric=['MI'],
                metric_weight=[1.0],
                radius_or_number_of_bins=[32],
                sampling_strategy=['Regular'],
                sampling_percentage=[0.25],
                convergence_threshold=[1e-6],
                convergence_window_size=[10],
                shrink_factors=[[4, 2, 1]],
                smoothing_sigmas=[[2, 1, 0]],
                sigma_units=['vox'],
                use_histogram_matching=False,
                winsorize_lower_quantile=0.005,
                winsorize_upper_quantile=0.995,
                initial_moving_transform_com=True,
                write_composite_transform=False,
                output_warped_image=True,
                interpolation='Linear',
            ),
            name='ants_pet_coreg',
            n_procs=omp_nthreads,
            mem_gb=5,
        )
        coreg_target = 'target_file'
        coreg_output = 'out_reg_file'
    else:
        coreg = pe.Node(
            MRICoreg(dof=pet2anat_dof, sep=[4], ftol=0.0001, linmintol=0.01),
            name='mri_coreg',
            n_procs=omp_nthreads,
            mem_gb=5,
        )
        coreg_target = 'fixed_image'
        coreg_output = 'forward_transforms'
    convert_xfm = pe.Node(ConcatenateXFMs(inverse=True), name='convert_xfm')

    connections = [
        (
            inputnode,
            mask_brain,
            [
                ('anat_preproc', 'in_file'),
                ('anat_mask', 'in_mask'),
            ],
        ),
        (
            inputnode,
            coreg,
            [
                (
                    'ref_pet_brain',
                    'moving_image' if use_robust_register else 'source_file',
                )
            ],
        ),
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
    ]

    if use_robust_register:
        connections.append((inputnode, coreg, [('anat_mask', 'fixed_image_masks')]))

    workflow.connect(connections)  # fmt:skip

    return workflow
