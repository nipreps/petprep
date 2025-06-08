import nibabel as nb
import numpy as np
import pandas as pd
from nipype.pipeline import engine as pe

from petprep.interfaces.tacs import ExtractTACs


def test_extract_tacs(tmp_path):
    pet_data = np.zeros((2, 2, 2, 2), dtype=float)
    seg = np.array(
        [
            [[1, 1], [2, 2]],
            [[1, 1], [2, 2]],
        ]
    )
    pet_data[..., 0] = seg
    pet_data[..., 1] = seg * 2
    pet_file = tmp_path / 'pet.nii.gz'
    seg_file = tmp_path / 'seg.nii.gz'
    dseg_tsv = tmp_path / 'dseg.tsv'

    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)
    nb.Nifti1Image(seg, np.eye(4)).to_filename(seg_file)
    pd.DataFrame({'index': [1, 2], 'name': ['region1', 'region2']}).to_csv(
        dseg_tsv, sep='\t', index=False
    )

    node = pe.Node(
        ExtractTACs(metadata={'FrameTimesStart': [0, 1], 'FrameDuration': [1, 1]}),
        name='tacs',
        base_dir=tmp_path,
    )
    node.inputs.pet_file = str(pet_file)
    node.inputs.segmentation = str(seg_file)
    node.inputs.dseg_tsv = str(dseg_tsv)
    res = node.run()

    out_df = pd.read_csv(res.outputs.out_file, sep='\t')
    assert list(out_df.columns) == ['FrameTimeStart', 'FrameTimesEnd', 'region1', 'region2']
    assert np.allclose(out_df['region1'], [1, 2])
    assert np.allclose(out_df['region2'], [2, 4])
