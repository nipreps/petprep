"""Interfaces for cropping PET images."""

from __future__ import annotations

import numpy as np
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, Undefined, traits
from nipype.utils.filemanip import fname_presuffix


def _voxel_size_z_mm(aff: np.ndarray) -> float:
    return float(np.linalg.norm(aff[:3, 2]))


def _k_increases_to_superior(aff: np.ndarray) -> bool:
    """
    Determine whether increasing k (z index) moves toward +Superior in world coords.

    We use the affine's 3rd column (voxel k-axis direction in world), and look at its
    world-Z component (RAS+ or LPS+ both have Superior as +Z in nibabel world).
    """
    k_dir_world = aff[:3, 2]
    return float(k_dir_world[2]) > 0


def _find_head_start_slice(mask3d: np.ndarray, from_superior: bool, min_vox_per_slice: int) -> int:
    """
    Find first slice index (k) from the superior end that contains >= min_vox_per_slice mask voxels.
    Returns the slice index in original k coordinates.
    """
    area = mask3d.sum(axis=(0, 1)).astype(np.int64)
    z_len = area.size

    if from_superior:
        for kk in range(z_len - 1, -1, -1):
            if area[kk] >= min_vox_per_slice:
                return kk
    else:
        for kk in range(0, z_len):
            if area[kk] >= min_vox_per_slice:
                return kk

    raise ValueError('No slice meets min_vox_per_slice; mask might be empty or too thin.')


class CropPetFromHeadFixedZInputSpec(TraitedSpec):
    """Input specification for :class:`CropPetFromHeadFixedZ`."""

    in_file = File(exists=True, mandatory=True, desc='Input PET NIfTI (3D or 4D)')
    mask_file = File(exists=True, mandatory=True, desc='Mask aligned to the PET grid (3D)')
    z_mm = traits.Float(200.0, usedefault=True, desc='Fixed z-extent to retain, in millimeters')
    pad_mm = traits.Float(20.0, usedefault=True, desc='Padding applied to both ends, in millimeters')
    min_vox_per_slice = traits.Int(
        1,
        usedefault=True,
        desc='Minimum mask voxels per slice to define the superior head slice',
    )
    out_file = File(desc='Output cropped PET image')
    out_mask = File(desc='Optional output cropped mask')


class CropPetFromHeadFixedZOutputSpec(TraitedSpec):
    """Output specification for :class:`CropPetFromHeadFixedZ`."""

    out_file = File(exists=True, desc='Cropped PET image')
    out_mask = File(desc='Cropped mask (if requested)', allow_none=True)


class CropPetFromHeadFixedZ(SimpleInterface):
    """Crop a PET image in the superior-inferior direction."""

    input_spec = CropPetFromHeadFixedZInputSpec
    output_spec = CropPetFromHeadFixedZOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        pet_img = nb.load(self.inputs.in_file)
        pet_data = np.asanyarray(pet_img.dataobj)
        if pet_data.ndim not in (3, 4):
            raise ValueError(f'PET must be 3D or 4D, got shape {pet_data.shape}')

        mask_img = nb.load(self.inputs.mask_file)
        mask3d = np.asanyarray(mask_img.dataobj).astype(np.uint8) > 0
        if mask3d.ndim != 3:
            raise ValueError('Mask must be 3D')
        if mask3d.shape != pet_img.shape[:3]:
            raise ValueError(
                f'Mask shape {mask3d.shape} does not match PET spatial shape {pet_img.shape[:3]}'
            )

        aff = pet_img.affine
        vz = _voxel_size_z_mm(aff)
        k_to_sup = _k_increases_to_superior(aff)
        superior_is_high_k = k_to_sup
        head_k = _find_head_start_slice(
            mask3d, from_superior=superior_is_high_k, min_vox_per_slice=self.inputs.min_vox_per_slice
        )

        nz = int(np.round(self.inputs.z_mm / vz))
        pad = int(np.ceil(self.inputs.pad_mm / vz)) if self.inputs.pad_mm > 0 else 0
        z_len = pet_img.shape[2]

        if superior_is_high_k:
            z1 = head_k + 1 + pad
            z0 = z1 - (nz + 2 * pad)
        else:
            z0 = head_k - pad
            z1 = z0 + (nz + 2 * pad)

        z0 = max(0, z0)
        z1 = min(z_len, z1)
        if z1 <= z0 + 1:
            raise ValueError(f'Computed invalid crop: z0={z0}, z1={z1}, Z={z_len}')

        aff2 = aff.copy()
        aff2[:3, 3] = aff[:3, 3] + aff[:3, 2] * float(z0)

        if pet_data.ndim == 3:
            pet_out_data = np.asarray(pet_data[:, :, z0:z1], dtype=np.float32)
        else:
            pet_out_data = np.asarray(pet_data[:, :, z0:z1, :], dtype=np.float32)

        pet_hdr2 = pet_img.header.copy()
        pet_hdr2.set_data_shape(pet_out_data.shape)
        pet_out_img = pet_img.__class__(pet_out_data, aff2, header=pet_hdr2)

        out_file = self.inputs.out_file
        if not out_file:
            out_file = fname_presuffix(self.inputs.in_file, suffix='_zcrop', newpath=runtime.cwd)
        pet_out_img.to_filename(out_file)
        self._results['out_file'] = out_file

        out_mask_path = None
        if self.inputs.out_mask:
            mask_out_data = mask3d[:, :, z0:z1].astype(np.uint8)
            mask_out_img = mask_img.__class__(mask_out_data, aff2)
            mask_out_img.set_data_dtype(np.uint8)
            mask_out_img.header['scl_slope'] = 1
            mask_out_img.header['scl_inter'] = 0
            mask_out_img.header['cal_min'] = 0
            mask_out_img.header['cal_max'] = 1
            mask_out_img.to_filename(self.inputs.out_mask)
            out_mask_path = self.inputs.out_mask

        self._results['out_mask'] = out_mask_path if out_mask_path is not None else Undefined
        return runtime
