import nibabel as nb
import numpy as np
from scipy.ndimage import label

from ..confounds import _smooth_binarize


def test_smooth_binarize_largest(tmp_path):
    data = np.zeros((5, 5, 5))
    data[1:3, 1:3, 1:3] = 1
    data[4, 4, 4] = 1
    img = nb.Nifti1Image(data, np.eye(4))
    src = tmp_path / 'input.nii.gz'
    img.to_filename(src)

    out = _smooth_binarize(str(src), fwhm=0.0, thresh=50.0)
    result = nb.load(out).get_fdata()
    _, num = label(result > 0)
    assert num == 1


def test_smooth_binarize_robust_range(tmp_path):
    data = np.zeros((5, 5, 5))
    data[:2, :2, :2] = 1  # Eight voxels at intensity 1
    data[3:, 3:, 3:] = 2  # Eight voxels at intensity 2
    img = nb.Nifti1Image(data, np.eye(4))
    src = tmp_path / 'robust_input.nii.gz'
    img.to_filename(src)

    out = _smooth_binarize(str(src), fwhm=0.0, thresh=50.0, use_robust_range=True)
    mask = nb.load(out).get_fdata()

    # With robust range, threshold should fall midway between 2nd and 98th percentiles (0 and 2)
    expected_mask = data == 2
    assert np.array_equal(mask, expected_mask)


def test_smooth_binarize_robust_range_collapse(tmp_path):
    data = np.zeros((3, 3, 3))
    img = nb.Nifti1Image(data, np.eye(4))
    src = tmp_path / 'collapse_input.nii.gz'
    img.to_filename(src)

    out = _smooth_binarize(str(src), fwhm=0.0, thresh=20.0, use_robust_range=True)
    mask = nb.load(out).get_fdata()

    # When the robust range collapses, threshold should fall back to max-based behavior (all zeros)
    assert not mask.any()


def test_smooth_binarize_max_threshold(tmp_path):
    data = np.zeros((5, 5, 5))
    data[1:4, 1:4, 1:4] = 2
    img = nb.Nifti1Image(data, np.eye(4))
    src = tmp_path / 'max_input.nii.gz'
    img.to_filename(src)

    out = _smooth_binarize(str(src), fwhm=0.0, thresh=0.25, use_robust_range=False)
    mask = nb.load(out).get_fdata()

    expected_mask = data > 0.5  # thresh * max -> 0.25 * 2
    assert np.array_equal(mask, expected_mask)
