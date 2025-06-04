import numpy as np
import nibabel as nb

from petprep.utils.misc import estimate_pet_mem_usage


def test_estimate_pet_mem_usage(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 10)), np.eye(4))
    pet_file = tmp_path / "pet.nii.gz"
    img.to_filename(pet_file)

    tlen, mem = estimate_pet_mem_usage(str(pet_file))
    size = 8 * np.prod(img.shape) / (1024 ** 3)
    assert tlen == 10
    assert np.isclose(mem['filesize'], size)
    assert np.isclose(mem['resampled'], size * 4)
    assert np.isclose(mem['largemem'], size * (max(tlen / 100, 1.0) + 4))