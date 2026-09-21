"""Direct PET-to-template ANTs registration, following CATNIP's candidate strategy."""

from nipype.interfaces import utility as niu
from nipype.interfaces.ants import Registration
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ...utils.pet_registration import (
    REGISTRATION_MASK_DILATION_MM,
    registration_initializations,
    registration_masks,
    select_registration,
)


def registration_parameters(sloppy=False):
    """The numerical ANTs recipe, also recorded for derivative reuse."""
    # Optimize SyN only at 8/4/2. The zero-iteration full-grid level represents
    # the resulting field on the fixed image grid for downstream resampling.
    return {
        'dimension': 3,
        'float': False,
        'random_seed': 14,
        'transforms': ['Rigid', 'Affine', 'SyN'],
        'transform_parameters': [(0.1,), (0.1,), (0.1, 3.0, 0.5)],
        'metric': ['MI'] * 3,
        'metric_weight': [1.0] * 3,
        'radius_or_number_of_bins': [32] * 3,
        'sampling_strategy': ['Regular'] * 3,
        'sampling_percentage': [0.25] * 3,
        'number_of_iterations': (
            [[20, 10, 5, 0]] * 3 if sloppy else [[1000, 500, 250, 100]] * 2 + [[200, 100, 50, 0]]
        ),
        'convergence_threshold': [1e-6] * 3,
        'convergence_window_size': [10] * 3,
        'shrink_factors': [[8, 4, 2, 1]] * 3,
        'smoothing_sigmas': [[3, 2, 1, 0]] * 3,
        'sigma_units': ['vox', 'vox', 'mm'],
        'use_histogram_matching': False,
        'winsorize_lower_quantile': 0.005,
        'winsorize_upper_quantile': 0.995,
        'write_composite_transform': True,
        'collapse_output_transforms': True,
        'output_transform_prefix': 'pet2template_',
        'output_warped_image': 'warped.nii.gz',
        'interpolation': 'Linear',
    }


def init_pet_template_reg_wf(*, omp_nthreads=1, sloppy=False, name='pet_template_reg_wf'):
    """Estimate one forward/inverse nonlinear pair from a PET reference to a template.

    The recipe follows CATNIP's MI rigid/affine/SyN strategy, with more
    conservative nonlinear regularization and spatial scales, without
    histogram matching. Only SyN is restricted to a template brain mask dilated
    by 5 mm. The original mask is used for initialization and candidate comparison.
    Indeterminate principal axes are
    omitted. Unlike CATNIP's runner, failure of an ANTs candidate fails this
    draft workflow, and affine plausibility gates are not yet implemented.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = (
        'The PET reference was normalized directly to a standard template using '
        'ANTs rigid, affine, and SyN registration with mutual information. '
        'Following CATNIP, header, center-of-mass, and, when determined, '
        'principal-axis initializations were compared on the template brain mask. '
        'The SyN metric was restricted to the template brain mask dilated by '
        f'{REGISTRATION_MASK_DILATION_MM:g} mm in physical space; rigid and affine '
        'stages were unmasked. '
        'SyN used a total-field smoothing variance of 0.5 in voxel units, '
        'optimization at shrink factors 8, 4, and 2 with image smoothing of '
        '3, 2, and 1 mm, and a final zero-iteration level on the fixed image grid. '
        'This PET-only normalization implementation is experimental.'
    )
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['petref', 'template', 'template_mask']), name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['petref2template_xfm', 'template2petref_xfm', 'warped_petref', 'report']
        ),
        name='outputnode',
    )
    imports = ['from pathlib import Path', 'import nibabel as nb', 'import numpy as np']
    initialize = pe.Node(
        niu.Function(
            function=registration_initializations,
            input_names=['fixed_image', 'moving_image', 'fixed_mask'],
            output_names=['transforms', 'names'],
            imports=imports,
        ),
        name='initialize',
    )
    dilate_mask = pe.Node(
        niu.Function(
            function=registration_masks,
            input_names=['fixed_mask', 'radius_mm'],
            output_names=['fixed_image_masks'],
        ),
        name='dilate_mask',
    )
    dilate_mask.inputs.radius_mm = REGISTRATION_MASK_DILATION_MM
    register = pe.MapNode(
        Registration(num_threads=omp_nthreads, **registration_parameters(sloppy)),
        iterfield=['initial_moving_transform'],
        name='register',
        n_procs=omp_nthreads,
        mem_gb=2,
    )
    select = pe.Node(
        niu.Function(
            function=select_registration,
            input_names=[
                'fixed_image',
                'fixed_mask',
                'warped_images',
                'forward',
                'inverse',
                'names',
            ],
            output_names=['forward', 'inverse', 'warped', 'report'],
            imports=imports,
        ),
        name='select',
    )
    workflow.connect(
        [
            (inputnode, register, [('petref', 'moving_image'), ('template', 'fixed_image')]),
            (inputnode, dilate_mask, [('template_mask', 'fixed_mask')]),
            (dilate_mask, register, [('fixed_image_masks', 'fixed_image_masks')]),
            (initialize, register, [('transforms', 'initial_moving_transform')]),
            (initialize, select, [('names', 'names')]),
            (inputnode, select, [('template', 'fixed_image'), ('template_mask', 'fixed_mask')]),
            (
                register,
                select,
                [
                    ('composite_transform', 'forward'),
                    ('inverse_composite_transform', 'inverse'),
                    ('warped_image', 'warped_images'),
                ],
            ),
            (
                select,
                outputnode,
                [
                    ('forward', 'petref2template_xfm'),
                    ('inverse', 'template2petref_xfm'),
                    ('warped', 'warped_petref'),
                    ('report', 'report'),
                ],
            ),
        ]
    )
    workflow.connect(inputnode, 'petref', initialize, 'moving_image')
    workflow.connect(inputnode, 'template', initialize, 'fixed_image')
    workflow.connect(inputnode, 'template_mask', initialize, 'fixed_mask')
    return workflow
