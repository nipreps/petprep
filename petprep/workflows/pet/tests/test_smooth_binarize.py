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

    out = _smooth_binarize(str(src), fwhm=0.0, thresh=0.5)
    result = nb.load(out).get_fdata()
    _, num = label(result > 0)
    assert num == 1
