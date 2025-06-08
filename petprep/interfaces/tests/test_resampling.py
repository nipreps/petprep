import nibabel as nb
import numpy as np
from nipype.pipeline import engine as pe

from petprep.interfaces.resampling import ResampleSeries


def test_resample_series_identity(tmp_path):
    data = np.arange(8, dtype=np.float32).reshape((2, 2, 2))
    img = nb.Nifti1Image(data, np.eye(4))
    ref = nb.Nifti1Image(np.zeros_like(data), np.eye(4))

    in_file = tmp_path / 'input.nii.gz'
    ref_file = tmp_path / 'ref.nii.gz'
    img.to_filename(in_file)
    ref.to_filename(ref_file)

    node = pe.Node(ResampleSeries(), name='resample', base_dir=tmp_path)
    node.inputs.in_file = str(in_file)
    node.inputs.ref_file = str(ref_file)
    node.inputs.transforms = []
    result = node.run()

    assert result.outputs.out_file == str(tmp_path / 'resample' / 'inputresampled.nii.gz')
    out_data = nb.load(result.outputs.out_file).get_fdata()
    assert np.allclose(out_data, data)