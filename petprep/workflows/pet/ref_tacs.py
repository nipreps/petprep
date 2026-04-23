from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe

from ...interfaces import ExtractRefTAC


def resample_pet_to_mask(pet_file, mask_file):
    import os

    import nibabel as nb
    import numpy as np
    from nilearn.image import resample_to_img
    from nilearn.image.resampling import BoundingBoxError

    pet_img = nb.load(pet_file)
    mask_img = nb.load(mask_file)

    try:
        resampled = resample_to_img(pet_img, mask_img, interpolation='continuous')
    except BoundingBoxError:
        from nibabel.processing import resample_from_to

        # Fallback to nibabel resampling, which is more permissive when
        # world-space bounding boxes are numerically inconsistent.
        if pet_img.ndim == 3:
            resampled = resample_from_to(pet_img, mask_img, order=1, mode='constant', cval=0.0)
        else:
            frame_data = []
            for frame_idx in range(pet_img.shape[-1]):
                frame_img = nb.Nifti1Image(
                    pet_img.dataobj[..., frame_idx],
                    pet_img.affine,
                    pet_img.header,
                )
                resampled_frame = resample_from_to(
                    frame_img,
                    mask_img,
                    order=1,
                    mode='constant',
                    cval=0.0,
                )
                frame_data.append(resampled_frame.get_fdata())
            resampled = nb.Nifti1Image(
                np.asarray(np.stack(frame_data, axis=-1), dtype=np.float32),
                mask_img.affine,
                mask_img.header,
            )

    out_file = os.path.abspath('pet_resampled.nii.gz')
    resampled.to_filename(out_file)
    return out_file


def init_pet_ref_tacs_wf(*, name: str = 'pet_ref_tacs_wf') -> pe.Workflow:
    """Extract reference region time activity curve."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_anat', 'mask_file', 'metadata', 'ref_mask_name']),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    resample_pet = pe.Node(
        Function(
            input_names=['pet_file', 'mask_file'],
            output_names=['resampled_pet'],
            function=resample_pet_to_mask,
        ),
        name='resample_pet',
    )

    tac = pe.Node(ExtractRefTAC(), name='tac')

    workflow.connect(
        [
            (
                inputnode,
                resample_pet,
                [('pet_anat', 'pet_file'), ('mask_file', 'mask_file')],
            ),
            (resample_pet, tac, [('resampled_pet', 'in_file')]),
            (
                inputnode,
                tac,
                [
                    ('mask_file', 'mask_file'),
                    ('metadata', 'metadata'),
                    ('ref_mask_name', 'ref_mask_name'),
                ],
            ),
            (tac, outputnode, [('out_file', 'timeseries')]),
        ]
    )

    return workflow


__all__ = ('init_pet_ref_tacs_wf',)
