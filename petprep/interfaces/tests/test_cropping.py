from __future__ import annotations

from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.interfaces.base import Undefined

from petprep.interfaces.cropping import CropPetFromHeadFixedZ


def test_crop_pet_from_head_superior_high_k(tmp_path: Path):
    """Cropping should trim slices inferior to the head when k increases superiorly."""

    data = np.stack([np.full((2, 2), fill_value=z, dtype=np.float32) for z in range(6)], axis=-1)
    pet_img = nb.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    pet_file = tmp_path / 'pet.nii.gz'
    pet_img.to_filename(pet_file)

    mask = np.zeros((2, 2, 6), dtype=np.uint8)
    mask[..., 4:] = 1
    mask_file = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(mask, pet_img.affine).to_filename(mask_file)

    result = CropPetFromHeadFixedZ(
        in_file=str(pet_file),
        mask_file=str(mask_file),
        z_mm=4.0,
        pad_mm=0.0,
        min_vox_per_slice=1,
        min_mask_volume_ml=0.0,
    ).run()

    cropped = nb.load(result.outputs.out_file)
    assert cropped.shape == (2, 2, 2)
    assert np.allclose(cropped.get_fdata(), data[..., 4:])
    assert np.allclose(cropped.affine[:3, 3], [0.0, 0.0, 8.0])


def test_crop_pet_from_head_superior_low_k(tmp_path: Path):
    """Cropping should handle grids where superior is at low k indices."""

    data = np.stack([np.full((2, 2), fill_value=z, dtype=np.float32) for z in range(6)], axis=-1)
    # Flip z axis to make superior correspond to low k
    affine = np.diag([2.0, 2.0, -2.0, 1.0])
    pet_img = nb.Nifti1Image(data, affine)
    pet_file = tmp_path / 'pet_inferior.nii.gz'
    pet_img.to_filename(pet_file)

    mask = np.zeros((2, 2, 6), dtype=np.uint8)
    mask[..., :2] = 1
    mask_file = tmp_path / 'mask_inferior.nii.gz'
    nb.Nifti1Image(mask, affine).to_filename(mask_file)

    out_mask = tmp_path / 'mask_crop.nii.gz'
    result = CropPetFromHeadFixedZ(
        in_file=str(pet_file),
        mask_file=str(mask_file),
        z_mm=4.0,
        pad_mm=0.0,
        min_vox_per_slice=1,
        min_mask_volume_ml=0.0,
        out_mask=str(out_mask),
    ).run()

    cropped = nb.load(result.outputs.out_file)
    assert cropped.shape == (2, 2, 2)
    assert np.allclose(cropped.get_fdata(), data[..., :2])
    assert np.allclose(cropped.affine[:3, 3], [0.0, 0.0, 0.0])

    cropped_mask = nb.load(result.outputs.out_mask)
    assert cropped_mask.shape == (2, 2, 2)
    assert np.array_equal(cropped_mask.get_fdata(), mask[..., :2])


def test_crop_pet_from_head_small_mask_returns_original(tmp_path: Path):
    """Cropping should be skipped when the mask volume is too small."""

    data = np.ones((2, 2, 2), dtype=np.float32)
    pet_img = nb.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    pet_file = tmp_path / 'pet_small.nii.gz'
    pet_img.to_filename(pet_file)

    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1  # 1 voxel -> 8 mm^3 -> 0.008 mL
    mask_file = tmp_path / 'mask_small.nii.gz'
    nb.Nifti1Image(mask, pet_img.affine).to_filename(mask_file)

    result = CropPetFromHeadFixedZ(
        in_file=str(pet_file), mask_file=str(mask_file), min_mask_volume_ml=500.0
    ).run()

    assert result.outputs.out_file == str(pet_file)
    assert result.outputs.out_mask is Undefined
