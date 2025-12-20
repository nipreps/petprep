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


def test_combine_pet_runs_concatenates_runs(tmp_path: Path) -> None:
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


def test_combine_pet_runs_handles_3d_inputs(tmp_path: Path) -> None:
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
