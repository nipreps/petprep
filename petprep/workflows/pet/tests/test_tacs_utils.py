import nibabel as nb
import numpy as np
import pandas as pd

from ..tacs import _extract_mask_tacs, _make_mask


def test_make_mask(tmp_path, monkeypatch):
    data = np.array([[[0, 1], [2, 0]], [[1, 2], [0, 0]]], dtype=np.int16)
    seg = nb.Nifti1Image(data, np.eye(4))
    seg_file = tmp_path / 'seg.nii.gz'
    seg.to_filename(seg_file)

    monkeypatch.chdir(tmp_path)
    out = _make_mask(str(seg_file), labels=[1, 2])
    expected = tmp_path / 'mask.nii.gz'
    assert out == str(expected.resolve())
    img = nb.load(out)
    mask = img.get_fdata()
    assert mask.shape == data.shape
    assert img.get_data_dtype() == np.uint8
    expected_mask = np.isin(data, [1, 2])
    assert np.array_equal(mask, expected_mask.astype(np.uint8))


def test_extract_mask_tacs(tmp_path, monkeypatch):
    frame0 = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    frame1 = np.arange(8, 16, dtype=np.float32).reshape(2, 2, 2)
    pet_data = np.stack([frame0, frame1], axis=-1)
    pet_img = nb.Nifti1Image(pet_data, np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    pet_img.to_filename(pet_file)

    mask_data = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 0]]], dtype=np.uint8)
    mask_img = nb.Nifti1Image(mask_data, np.eye(4))
    mask_file = tmp_path / 'mask.nii.gz'
    mask_img.to_filename(mask_file)

    meta = {'FrameTimesStart': [0, 1], 'FrameDuration': [1, 1]}
    monkeypatch.chdir(tmp_path)
    out = _extract_mask_tacs(str(pet_file), str(mask_file), meta, name='mask')
    expected = tmp_path / 'mask_tacs.tsv'
    assert out == str(expected.resolve())

    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['FrameTimeStart', 'FrameTimesEnd', 'mask']

    flat_pet = pet_data.reshape(-1, 2)
    vals = flat_pet[mask_data.reshape(-1).astype(bool)]
    expected_means = vals.mean(axis=0)

    assert df['FrameTimeStart'].tolist() == [0, 1]
    assert df['FrameTimesEnd'].tolist() == [1, 2]
    assert np.allclose(df['mask'].values, expected_means)