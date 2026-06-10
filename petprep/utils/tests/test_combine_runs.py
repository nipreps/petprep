from __future__ import annotations

import json
from pathlib import Path

import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout

from petprep.utils.bids import combine_pet_runs


def _write_nifti(path: Path, data: np.ndarray) -> None:
    img = nb.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(path)


def _write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=4))


def _write_dataset_description(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'Name': 'Test dataset', 'BIDSVersion': '1.8.0'}))


def test_combine_pet_runs_concatenates_runs(tmp_path: Path, monkeypatch) -> None:
    bids_dir = tmp_path / 'bids'
    dataset_description = bids_dir / 'dataset_description.json'
    _write_dataset_description(dataset_description)

    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'

    _write_nifti(run1_img, np.ones((2, 2, 2, 2)))
    _write_nifti(run2_img, np.full((2, 2, 2, 1), 2))

    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0, 1.0], 'FrameDuration': [1.0, 1.0]},
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0], 'FrameDuration': [2.0]},
    )

    layout = BIDSLayout(bids_dir, validate=False)

    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    combined_dir, combined_files = combine_pet_runs(
        bids_dir=bids_dir,
        layout=layout,
        work_dir=tmp_path / 'work',
        subjects=['01'],
        bids_filters={},
    )

    expected_img = combined_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_pet.nii.gz'
    expected_json = expected_img.with_suffix('').with_suffix('.json')

    assert combined_files == [str(expected_img)]
    assert expected_img.exists()
    assert expected_json.exists()

    combined_img = nb.load(expected_img)
    assert combined_img.shape == (2, 2, 2, 3)
    data = combined_img.get_fdata()
    assert np.all(data[..., 0:2] == 1)
    assert np.all(data[..., 2] == 2)

    combined_meta = json.loads(expected_json.read_text())
    assert combined_meta['FrameTimesStart'] == [0.0, 1.0, 2.0]
    assert combined_meta['FrameDuration'] == [1.0, 1.0, 2.0]
    assert combined_meta['AcquisitionDuration'] == 4.0

    run_sources = list(combined_dir.glob('**/*run-*'))
    assert run_sources == []


def test_combine_pet_runs_handles_3d_inputs(tmp_path: Path, monkeypatch) -> None:
    bids_dir = tmp_path / 'bids'
    dataset_description = bids_dir / 'dataset_description.json'
    _write_dataset_description(dataset_description)

    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'

    _write_nifti(run1_img, np.ones((2, 2, 2)))
    _write_nifti(run2_img, np.full((2, 2, 2), 2))

    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0], 'FrameDuration': [2.0]},
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [2.0], 'FrameDuration': [3.0]},
    )

    layout = BIDSLayout(bids_dir, validate=False)

    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    combined_dir, combined_files = combine_pet_runs(
        bids_dir=bids_dir,
        layout=layout,
        work_dir=tmp_path / 'work',
        subjects=['01'],
        bids_filters={},
    )

    expected_img = combined_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_pet.nii.gz'
    expected_json = expected_img.with_suffix('').with_suffix('.json')

    assert combined_files == [str(expected_img)]
    assert expected_img.exists()
    assert expected_json.exists()

    combined_img = nb.load(expected_img)
    assert combined_img.shape == (2, 2, 2, 2)
    data = combined_img.get_fdata()
    assert np.all(data[..., 0] == 1)
    assert np.all(data[..., 1] == 2)

    combined_meta = json.loads(expected_json.read_text())
    assert combined_meta['FrameTimesStart'] == [0.0, 2.0]
    assert combined_meta['FrameDuration'] == [2.0, 3.0]
    assert combined_meta['AcquisitionDuration'] == 5.0


def test_combine_pet_runs_handles_mixed_dimensions(tmp_path: Path, monkeypatch) -> None:
    bids_dir = tmp_path / 'bids'
    dataset_description = bids_dir / 'dataset_description.json'
    _write_dataset_description(dataset_description)

    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'

    _write_nifti(run1_img, np.ones((2, 2, 2)))
    _write_nifti(run2_img, np.stack([np.full((2, 2, 2), 2), np.full((2, 2, 2), 3)], axis=-1))

    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0], 'FrameDuration': [5.0]},
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0, 2.0], 'FrameDuration': [1.0, 1.5]},
    )

    layout = BIDSLayout(bids_dir, validate=False)

    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    combined_dir, combined_files = combine_pet_runs(
        bids_dir=bids_dir,
        layout=layout,
        work_dir=tmp_path / 'work',
        subjects=['01'],
        bids_filters={},
    )

    expected_img = combined_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_pet.nii.gz'
    expected_json = expected_img.with_suffix('').with_suffix('.json')

    assert combined_files == [str(expected_img)]
    assert expected_img.exists()
    assert expected_json.exists()

    combined_img = nb.load(expected_img)
    assert combined_img.shape == (2, 2, 2, 3)
    data = combined_img.get_fdata()
    assert np.all(data[..., 0] == 1)
    assert np.all(data[..., 1] == 2)
    assert np.all(data[..., 2] == 3)

    combined_meta = json.loads(expected_json.read_text())
    assert combined_meta['FrameTimesStart'] == [0.0, 5.0, 7.0]
    assert combined_meta['FrameDuration'] == [5.0, 1.0, 1.5]
    assert combined_meta['AcquisitionDuration'] == 7.5


def test_combine_pet_runs_aligns_runs_to_common_timezero(
    tmp_path: Path, monkeypatch
) -> None:
    bids_dir = tmp_path / 'bids'
    dataset_description = bids_dir / 'dataset_description.json'
    _write_dataset_description(dataset_description)

    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'

    _write_nifti(run1_img, np.ones((2, 2, 2, 2)))
    _write_nifti(run2_img, np.full((2, 2, 2, 2), 2))

    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {
            'TimeZero': '11:19:37',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0, 600.0],
            'FrameDuration': [600.0, 105.552],
            'FrameReferenceTime': [300.0, 652.776],
        },
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {
            'TimeZero': '13:27:58',
            'InjectionStart': -7701.0,
            'FrameTimesStart': [0.0, 600.0],
            'FrameDuration': [600.0, 101.016],
            'FrameReferenceTime': [300.0, 650.508],
        },
    )

    layout = BIDSLayout(bids_dir, validate=False)

    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    combined_dir, _ = combine_pet_runs(
        bids_dir=bids_dir,
        layout=layout,
        work_dir=tmp_path / 'work',
        subjects=['01'],
        bids_filters={},
    )

    expected_json = (
        combined_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_pet.json'
    )
    combined_meta = json.loads(expected_json.read_text())

    assert combined_meta['TimeZero'] == '11:19:37'
    assert combined_meta['InjectionStart'] == 0.0
    assert combined_meta['FrameTimesStart'] == [0.0, 600.0, 7701.0, 8301.0]
    assert combined_meta['FrameDuration'] == [600.0, 105.552, 600.0, 101.016]
    assert combined_meta['FrameReferenceTime'] == [300.0, 652.776, 8001.0, 8351.508]


def test_combine_pet_runs_rescales_decay_corrected_runs(
    tmp_path: Path, monkeypatch
) -> None:
    bids_dir = tmp_path / 'bids'
    dataset_description = bids_dir / 'dataset_description.json'
    _write_dataset_description(dataset_description)

    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'

    _write_nifti(run1_img, np.ones((2, 2, 2, 1)))
    _write_nifti(run2_img, np.full((2, 2, 2, 1), 2))

    common_metadata = {
        'ImageDecayCorrected': True,
        'ImageDecayCorrectionTime': 0.0,
        'RadionuclideHalfLife': 10.0,
        'FrameDuration': [10.0],
        'DecayCorrectionFactor': [1.0],
    }
    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {
            **common_metadata,
            'TimeZero': '00:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0],
        },
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {
            **common_metadata,
            'TimeZero': '00:00:10',
            'InjectionStart': -10.0,
            'FrameTimesStart': [0.0],
        },
    )

    layout = BIDSLayout(bids_dir, validate=False)

    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    combined_dir, _ = combine_pet_runs(
        bids_dir=bids_dir,
        layout=layout,
        work_dir=tmp_path / 'work',
        subjects=['01'],
        bids_filters={},
    )

    expected_img = combined_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_pet.nii.gz'
    expected_json = expected_img.with_suffix('').with_suffix('.json')

    combined_img = nb.load(expected_img)
    data = combined_img.get_fdata()
    assert np.all(data[..., 0] == 1)
    assert np.allclose(data[..., 1], 4)

    combined_meta = json.loads(expected_json.read_text())
    assert combined_meta['FrameTimesStart'] == [0.0, 10.0]
    assert combined_meta['ImageDecayCorrectionTime'] == 0.0
    assert combined_meta['DecayCorrectionFactor'] == [1.0, 2.0]
