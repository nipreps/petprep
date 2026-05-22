import nibabel as nb
import nitransforms as nt
import numpy as np

from petprep.interfaces.resampling import resample_image


def test_resample_image_passes_coordinates_as_points():
    source = nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4))
    target = nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))

    resampled = resample_image(source, target, nt.Affine(), order=0)

    assert resampled.shape == target.shape
    assert np.allclose(resampled.get_fdata(), 1.0)
