import nibabel as nb
import numpy as np

from petprep.utils.misc import PET_RESAMPLE_MEMORY_SCALE, estimate_pet_mem_usage


def test_estimate_pet_mem_usage(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 10)), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    tlen, mem = estimate_pet_mem_usage(str(pet_file))
    size = 8 * np.prod(img.shape) / (1024**3)
    frame_size = 8 * np.prod(img.shape[:3]) / (1024**3)
    assert tlen == 10
    assert np.isclose(mem['filesize'], size)
    assert np.isclose(mem['frame'], frame_size)
    assert np.isclose(mem['reference'], max(size * 1.5, size + frame_size * 4))
    assert np.isclose(mem['resampled'], size * PET_RESAMPLE_MEMORY_SCALE)
    assert np.isclose(mem['largemem'], size * (max(tlen / 100, 1.0) + 4))


def test_estimate_pet_mem_usage_refreshes_overwritten_file(tmp_path):
    pet_file = tmp_path / 'pet.nii.gz'

    nb.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)).to_filename(pet_file)
    assert estimate_pet_mem_usage(str(pet_file))[0] == 1

    nb.Nifti1Image(np.zeros((5, 5, 5, 2), dtype=np.float32), np.eye(4)).to_filename(pet_file)
    assert estimate_pet_mem_usage(str(pet_file))[0] == 2
