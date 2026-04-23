from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe

from ...interfaces import ExtractRefTAC


def resample_mask_to_pet(mask_file, pet_file):
    import os

    import nibabel as nb
    import numpy as np
    from nilearn.image import resample_img, resample_to_img
    from nilearn.image.resampling import BoundingBoxError

    pet_img = nb.load(pet_file)
    mask_img = nb.load(mask_file)

    same_grid = mask_img.shape[:3] == pet_img.shape[:3] and np.allclose(
        mask_img.affine, pet_img.affine
    )

    if same_grid:
        resampled = mask_img
    else:
        try:
            resampled = resample_to_img(
                mask_file,
                pet_file,
                interpolation='nearest',
                copy_header=True,
            )
        except BoundingBoxError:
            try:
                resampled = resample_img(
                    mask_img,
                    target_affine=pet_img.affine,
                    target_shape=pet_img.shape[:3],
                    interpolation='nearest',
                    force_resample=True,
                    copy_header=True,
                )
            except BoundingBoxError:
                zeros = np.zeros(pet_img.shape[:3], dtype=np.int16)
                resampled = nb.Nifti1Image(zeros, pet_img.affine, pet_img.header)

    out_data = np.rint(resampled.get_fdata()).astype(np.int16)
    out_file = os.path.abspath('mask_resampled.nii.gz')
    nb.Nifti1Image(out_data, pet_img.affine, pet_img.header).to_filename(out_file)
    return out_file


def init_pet_ref_tacs_wf(*, name: str = 'pet_ref_tacs_wf') -> pe.Workflow:
    """Extract reference region time activity curve."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_anat', 'mask_file', 'metadata', 'ref_mask_name']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    resample_mask = pe.Node(
        Function(
            input_names=['mask_file', 'pet_file'],
            output_names=['resampled_mask'],
            function=resample_mask_to_pet,
        ),
        name='resample_mask',
    )

    tac = pe.Node(ExtractRefTAC(), name='tac')

    workflow.connect(
        [
            (
                inputnode,
                resample_mask,
                [('pet_anat', 'pet_file'), ('mask_file', 'mask_file')],
            ),
            (inputnode, tac, [('pet_anat', 'in_file')]),
            (resample_mask, tac, [('resampled_mask', 'mask_file')]),
            (
                inputnode,
                tac,
                [
                    ('metadata', 'metadata'),
                    ('ref_mask_name', 'ref_mask_name'),
                ],
            ),
            (tac, outputnode, [('out_file', 'timeseries')]),
        ]
    )

    return workflow


__all__ = ('init_pet_ref_tacs_wf',)
