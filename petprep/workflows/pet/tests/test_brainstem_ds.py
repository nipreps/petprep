from pathlib import Path

import nibabel as nb
import numpy as np
from ....interfaces import DerivativesDataSink


def test_brainstem_dseg_extension(tmp_path: Path):
    bids_root = tmp_path / 'bids'
    anat_dir = bids_root / 'sub-01' / 'anat'
    anat_dir.mkdir(parents=True, exist_ok=True)
    src_file = anat_dir / 'sub-01_T1w.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)).to_filename(src_file)

    in_img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    in_file = tmp_path / 'seg.nii.gz'
    in_img.to_filename(in_file)

    ds = DerivativesDataSink(
        base_directory=tmp_path,
        desc='brainstem',
        suffix='dseg',
        extension='.nii.gz',
        compress=True,
    )
    result = ds.run(in_file=str(in_file), source_file=str(src_file))
    assert Path(result.outputs.out_file).name.endswith('desc-brainstem_dseg.nii.gz')