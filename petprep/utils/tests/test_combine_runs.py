from __future__ import annotations

import json
from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from bids.layout import BIDSLayout

from petprep.utils import bids as bids_utils
from petprep.utils.bids import combine_pet_runs


def _write_nifti(path: Path, data: np.ndarray) -> None:
    img = nb.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(path)


def _write_metadata(path: Path, metadata: dict) -> None:
    metadata = {
        'ImageDecayCorrected': False,
        'ImageDecayCorrectionTime': 0.0,
        **metadata,
    }
    path.write_text(json.dumps(metadata, indent=4))


def _write_dataset_description(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'Name': 'Test dataset', 'BIDSVersion': '1.8.0'}))


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('11:19:37', 40777.0),
        (' 01:02:03.5 ', 3723.5),
        ('25:00:00', None),
        ('not-a-time', None),
        (None, None),
    ],
)
def test_parse_timezero(value, expected) -> None:
    assert bids_utils._parse_timezero(value) == expected


@pytest.mark.parametrize(
    ('base_meta', 'meta', 'expected'),
    [
        ({'InjectionStart': 0.0}, {'InjectionStart': -7701.0}, 7701.0),
        ({}, {'InjectionStart': -1.0}, None),
        ({'InjectionStart': 'bad'}, {'InjectionStart': -1.0}, None),
        ({'InjectionStart': -1.0}, {'InjectionStart': 0.0}, None),
    ],
)
def test_run_offset_from_injection_start(base_meta, meta, expected) -> None:
    assert bids_utils._run_offset_from_injection_start(base_meta, meta) == expected


@pytest.mark.parametrize(
    ('starts', 'durations', 'expected'),
    [
        ([0.0, 10.0], [10.0, 20.0], 30.0),
        ([5.0], [1.0, 2.0], 8.0),
        ([5.0], [], 5.0),
        ([], [10.0], 0.0),
    ],
)
def test_frame_end(starts, durations, expected) -> None:
    assert bids_utils._frame_end(starts, durations) == expected


def test_run_time_offsets_fallbacks() -> None:
    metas = [
        {'FrameTimesStart': [0.0], 'FrameDuration': [5.0]},
        {'FrameTimesStart': [0.0], 'FrameDuration': [2.0]},
        {'FrameTimesStart': [10.0], 'FrameDuration': [1.0]},
        {'FrameDuration': [3.0]},
    ]

    assert bids_utils._run_time_offsets(metas) == [0.0, 5.0, 0.0, 11.0]
    assert bids_utils._run_time_offsets_with_reliability(metas) == (
        [0.0, 5.0, 0.0, 11.0],
        [True, False, False, False],
    )


def test_run_time_offsets_uses_injection_start_when_timezero_is_missing() -> None:
    metas = [
        {'InjectionStart': 0.0, 'FrameTimesStart': [0.0], 'FrameDuration': [5.0]},
        {'InjectionStart': -10.0, 'FrameTimesStart': [0.0], 'FrameDuration': [2.0]},
    ]

    assert bids_utils._run_time_offsets(metas) == [0.0, 10.0]


def test_run_time_offsets_does_not_treat_timezero_alone_as_exact() -> None:
    metas = [
        {'TimeZero': '12:00:00', 'FrameTimesStart': [0.0], 'FrameDuration': [5.0]},
        {'TimeZero': '13:00:00', 'FrameTimesStart': [0.0], 'FrameDuration': [2.0]},
    ]

    assert bids_utils._run_time_offsets_with_reliability(metas) == (
        [0.0, 5.0],
        [True, False],
    )


def test_run_time_offsets_uses_injection_start_to_retain_elapsed_days() -> None:
    metas = [
        {
            'TimeZero': '08:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [5.0],
        },
        {
            'TimeZero': '09:00:00',
            'InjectionStart': -90000.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [2.0],
        },
    ]

    assert bids_utils._run_time_offsets_with_reliability(metas) == (
        [0.0, 90000.0],
        [True, True],
    )


def test_run_time_offsets_rejects_inconsistent_timing_references() -> None:
    metas = [
        {'TimeZero': '12:00:00', 'InjectionStart': 0.0},
        {'TimeZero': '14:00:00', 'InjectionStart': 0.0},
    ]

    with pytest.raises(ValueError, match='different injections or incompatible timing references'):
        bids_utils._run_time_offsets(metas)


def test_run_time_offsets_rejects_overlapping_runs() -> None:
    metas = [
        {
            'TimeZero': '12:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [10.0],
        },
        {
            'TimeZero': '12:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [10.0],
        },
    ]

    with pytest.raises(ValueError, match='frame times overlap'):
        bids_utils._run_time_offsets(metas)


@pytest.mark.parametrize(
    ('value', 'frame_count', 'expected'),
    [
        ([1.0, 2.0], 2, [1.0, 2.0]),
        ([1.0], 3, [1.0, 1.0, 1.0]),
        ([1.0, 2.0], 3, None),
        ('not-framewise', 1, None),
        ([1.0], 0, None),
    ],
)
def test_metadata_as_framewise(value, frame_count, expected) -> None:
    assert bids_utils._metadata_as_framewise(value, frame_count) == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('C11', 'C11'),
        ('11C', 'C11'),
        ('carbon-11', 'C11'),
        ('[11C]PIB', 'C11'),
        ('F18', 'F18'),
        ('18F', 'F18'),
        ('18Fluorine', 'F18'),
        ('fluorine-18', 'F18'),
        ('[18F]FDG', 'F18'),
        ('nitrogen-13', 'N13'),
        ('[15O]H2O', 'O15'),
        ('VAT', None),
        ('11C18F', None),
        (None, None),
    ],
)
def test_infer_radionuclide(value, expected) -> None:
    assert bids_utils._infer_radionuclide(value) == expected


@pytest.mark.parametrize(
    ('meta', 'expected'),
    [
        ({'RadionuclideHalfLife': 123.0, 'TracerRadionuclide': '18F'}, None),
        ({'RadionuclideHalfLife': 'bad', 'TracerRadionuclide': '18Fluorine'}, 6586.2),
        ({'RadionuclideHalfLife': 0.0, 'TracerRadionuclide': '11C'}, 1220.4),
        ({'TracerRadionuclide': 'O15'}, 122.24),
        ({'TracerRadionuclide': 'unknown'}, None),
    ],
)
def test_metadata_half_life(meta, expected) -> None:
    assert bids_utils._metadata_half_life(meta) == expected


@pytest.mark.parametrize(
    ('metas', 'run_offsets'),
    [
        ([], []),
        ([{'ImageDecayCorrected': False, 'ImageDecayCorrectionTime': 0.0}], [0.0]),
        ([{'ImageDecayCorrected': True}], [0.0]),
        ([{'ImageDecayCorrected': True, 'ImageDecayCorrectionTime': 'bad'}], [0.0]),
    ],
)
def test_absolute_decay_correction_times_rejects_invalid_metadata(metas, run_offsets) -> None:
    assert bids_utils._absolute_decay_correction_times(metas, run_offsets) is None


def test_decay_rescale_factors_rejects_ambiguous_metadata() -> None:
    assert bids_utils._decay_rescale_factors([], []) == []
    assert bids_utils._decay_rescale_factors(
        [{'ImageDecayCorrected': False, 'ImageDecayCorrectionTime': 0.0}], [0.0]
    ) == [1.0]

    with pytest.raises(ValueError, match='ImageDecayCorrected'):
        bids_utils._decay_rescale_factors([{}], [0.0])

    with pytest.raises(ValueError, match='ImageDecayCorrectionTime'):
        bids_utils._decay_rescale_factors([{'ImageDecayCorrected': False}], [0.0])

    corrected_missing_fields = [{'ImageDecayCorrected': True}]
    with pytest.raises(ValueError, match='ImageDecayCorrectionTime'):
        bids_utils._decay_rescale_factors(corrected_missing_fields, [0.0])

    invalid_half_life = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 0.0,
        }
    ]
    assert bids_utils._decay_rescale_factors(invalid_half_life, [0.0]) == [1.0]

    missing_run_decay_time = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
        },
        {'ImageDecayCorrected': True, 'RadionuclideHalfLife': 10.0},
    ]
    with pytest.raises(ValueError, match='ImageDecayCorrectionTime'):
        bids_utils._decay_rescale_factors(missing_run_decay_time, [0.0, 10.0])


@pytest.mark.parametrize('correction_time', [np.nan, np.inf, -np.inf])
def test_decay_rescale_factors_rejects_nonfinite_correction_times(correction_time) -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': correction_time,
        }
    ]

    with pytest.raises(ValueError, match='finite number'):
        bids_utils._decay_rescale_factors(metas, [0.0])


def test_decay_rescale_factors_rejects_nonfinite_absolute_correction_time() -> None:
    metas = [{'ImageDecayCorrected': True, 'ImageDecayCorrectionTime': 0.0}]

    with pytest.raises(ValueError, match='defined for every run'):
        bids_utils._decay_rescale_factors(metas, [np.inf])


def test_decay_rescale_factors() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
        },
    ]

    assert bids_utils._decay_rescale_factors(metas, [0.0, 10.0]) == pytest.approx([1.0, 2.0])


def test_decay_rescale_factors_uses_tracer_radionuclide_half_life() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'TracerRadionuclide': '18Fluorine',
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'TracerRadionuclide': '18F',
        },
    ]

    assert bids_utils._decay_rescale_factors(metas, [0.0, 6586.2]) == pytest.approx([1.0, 2.0])


def test_decay_rescale_factors_supports_oxygen_15() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'TracerRadionuclide': 'O15',
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'TracerRadionuclide': '15O',
        },
    ]

    assert bids_utils._decay_rescale_factors(metas, [0.0, 122.24]) == pytest.approx([1.0, 2.0])


def test_decay_rescale_factors_requires_matching_radionuclides() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'TracerRadionuclide': '18F',
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'TracerRadionuclide': '11C',
        },
    ]

    with pytest.raises(ValueError, match='inconsistent radionuclide half-lives'):
        bids_utils._decay_rescale_factors(metas, [0.0, 10.0])


@pytest.mark.parametrize('run_offsets', [[0.0, 2000.0], [2000.0, 0.0]])
def test_decay_rescale_factors_rejects_out_of_range_factors(run_offsets) -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 1.0,
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 1.0,
        },
    ]

    with pytest.raises(ValueError, match='outside the supported numeric range'):
        bids_utils._decay_rescale_factors(metas, run_offsets)


def test_merge_frame_metadata_rejects_unresolved_decay_correction_metadata() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [10.0],
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [10.0],
        },
    ]

    with pytest.raises(ValueError, match='radionuclide half-life'):
        bids_utils._merge_frame_metadata(
            metas,
            run_offsets=[0.0, 600.0],
            decay_rescale_factors=[1.0, 1.0],
        )


def test_merge_frame_metadata_rejects_incorrect_decay_rescale_factors() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
        },
    ]

    with pytest.raises(ValueError, match='do not match the PET decay metadata'):
        bids_utils._merge_frame_metadata(
            metas,
            run_offsets=[0.0, 10.0],
            decay_rescale_factors=[1.0, 1.0],
        )


def test_merge_frame_metadata_preserves_matching_decay_correction_metadata() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [10.0],
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': -600.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [10.0],
        },
    ]

    merged = bids_utils._merge_frame_metadata(
        metas,
        run_offsets=[0.0, 600.0],
        decay_rescale_factors=[1.0, 1.0],
    )

    assert merged['ImageDecayCorrected'] is True
    assert merged['ImageDecayCorrectionTime'] == 0.0


def test_merge_frame_metadata_merges_and_drops_framewise_metadata() -> None:
    metas = [
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
            'FrameTimesStart': [0.0],
            'VolumeTiming': [0.0],
            'FrameDuration': [1.0],
            'ScaleFactor': [1.0],
            'PromptRate': [10.0],
            'SinglesRate': [20.0],
            'RandomRate': [30.0],
            'ScatterFraction': [0.1],
            'DecayFactor': [1.0],
        },
        {
            'ImageDecayCorrected': True,
            'ImageDecayCorrectionTime': 0.0,
            'RadionuclideHalfLife': 10.0,
            'FrameTimesStart': [0.0, 1.0],
            'VolumeTiming': [0.0, 1.0],
            'FrameDuration': [1.0, 1.0],
            'ScaleFactor': [2.0],
            'PromptRate': [11.0, 12.0],
            'SinglesRate': [21.0, 22.0],
            'RandomRate': [31.0, 32.0],
            'ScatterFraction': [0.2, 0.3],
            'DecayFactor': [1.0, 1.5],
            'FrameReferenceTime': [0.5, 1.5],
        },
    ]

    merged = bids_utils._merge_frame_metadata(
        metas,
        run_offsets=[0.0, 10.0],
        decay_rescale_factors=[1.0, 2.0],
    )

    assert merged['FrameTimesStart'] == [0.0, 10.0, 11.0]
    assert merged['VolumeTiming'] == [0.0, 10.0, 11.0]
    assert merged['FrameDuration'] == [1.0, 1.0, 1.0]
    assert merged['ScaleFactor'] == [1.0, 2.0, 2.0]
    assert merged['PromptRate'] == [10.0, 11.0, 12.0]
    assert merged['SinglesRate'] == [20.0, 21.0, 22.0]
    assert merged['RandomRate'] == [30.0, 31.0, 32.0]
    assert merged['ScatterFraction'] == [0.1, 0.2, 0.3]
    assert merged['DecayFactor'] == [1.0, 2.0, 3.0]
    assert 'FrameReferenceTime' not in merged


def test_merge_frame_metadata_drops_unmergeable_volume_timing() -> None:
    metas = [
        {
            'ImageDecayCorrected': False,
            'ImageDecayCorrectionTime': 0.0,
            'VolumeTiming': [0.0],
            'FrameTimesStart': [0.0],
            'FrameDuration': [1.0],
        },
        {
            'ImageDecayCorrected': False,
            'ImageDecayCorrectionTime': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [1.0],
        },
    ]

    merged = bids_utils._merge_frame_metadata(
        metas,
        run_offsets=[0.0, 1.0],
    )

    assert 'VolumeTiming' not in merged


def test_merge_frame_metadata_drops_unmergeable_frame_times_without_durations() -> None:
    metas = [
        {
            'ImageDecayCorrected': False,
            'ImageDecayCorrectionTime': 0.0,
            'FrameTimesStart': [0.0],
        },
        {
            'ImageDecayCorrected': False,
            'ImageDecayCorrectionTime': 0.0,
        },
    ]

    merged = bids_utils._merge_frame_metadata(metas, run_offsets=[0.0, 1.0])

    assert 'FrameTimesStart' not in merged
    assert 'FrameDuration' not in merged
    assert 'AcquisitionDuration' not in merged


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
        {
            'TimeZero': '00:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0, 1.0],
            'FrameDuration': [1.0, 1.0],
        },
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {
            'TimeZero': '00:00:02',
            'InjectionStart': -2.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [2.0],
        },
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
    assert combined_meta['ImageDecayCorrected'] is False
    assert combined_meta['ImageDecayCorrectionTime'] == 0.0

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
        {
            'TimeZero': '00:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [2.0],
        },
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {
            'TimeZero': '00:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [2.0],
            'FrameDuration': [3.0],
        },
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
        {
            'TimeZero': '00:00:00',
            'InjectionStart': 0.0,
            'FrameTimesStart': [0.0],
            'FrameDuration': [5.0],
        },
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {
            'TimeZero': '00:00:05',
            'InjectionStart': -5.0,
            'FrameTimesStart': [0.0, 2.0],
            'FrameDuration': [1.0, 1.5],
        },
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


def test_combine_pet_runs_aligns_runs_to_common_timezero(tmp_path: Path, monkeypatch) -> None:
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

    expected_json = combined_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_pet.json'
    combined_meta = json.loads(expected_json.read_text())

    assert combined_meta['TimeZero'] == '11:19:37'
    assert combined_meta['InjectionStart'] == 0.0
    assert combined_meta['FrameTimesStart'] == [0.0, 600.0, 7701.0, 8301.0]
    assert combined_meta['FrameDuration'] == [600.0, 105.552, 600.0, 101.016]
    assert combined_meta['FrameReferenceTime'] == [300.0, 652.776, 8001.0, 8351.508]


def test_combine_pet_runs_rescales_decay_corrected_runs(tmp_path: Path, monkeypatch) -> None:
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


def test_combine_pet_runs_rescales_decay_corrected_3d_runs(tmp_path: Path, monkeypatch) -> None:
    bids_dir = tmp_path / 'bids'
    dataset_description = bids_dir / 'dataset_description.json'
    _write_dataset_description(dataset_description)

    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'

    _write_nifti(run1_img, np.ones((2, 2, 2)))
    _write_nifti(run2_img, np.full((2, 2, 2), 2))

    common_metadata = {
        'ImageDecayCorrected': True,
        'ImageDecayCorrectionTime': 0.0,
        'RadionuclideHalfLife': 10.0,
        'FrameDuration': [10.0],
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

    combined_img = nb.load(expected_img)
    data = combined_img.get_fdata()
    assert np.all(data[..., 0] == 1)
    assert np.allclose(data[..., 1], 4)


def test_combine_pet_runs_rejects_unresolved_timing(tmp_path: Path, monkeypatch) -> None:
    bids_dir = tmp_path / 'bids'
    _write_dataset_description(bids_dir / 'dataset_description.json')
    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'
    _write_nifti(run1_img, np.ones((2, 2, 2)))
    _write_nifti(run2_img, np.ones((2, 2, 2)))
    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0], 'FrameDuration': [10.0]},
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {'FrameTimesStart': [0.0], 'FrameDuration': [10.0]},
    )
    layout = BIDSLayout(bids_dir, validate=False)
    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    with pytest.raises(ValueError, match='exact timing offset'):
        combine_pet_runs(
            bids_dir=bids_dir,
            layout=layout,
            work_dir=tmp_path / 'work',
            subjects=['01'],
            bids_filters={},
        )


def test_combine_pet_runs_rejects_mixed_decay_correction(tmp_path: Path, monkeypatch) -> None:
    bids_dir = tmp_path / 'bids'
    _write_dataset_description(bids_dir / 'dataset_description.json')
    pet_dir = bids_dir / 'sub-01' / 'pet'
    run1_img = pet_dir / 'sub-01_task-rest_run-01_pet.nii.gz'
    run2_img = pet_dir / 'sub-01_task-rest_run-02_pet.nii.gz'
    _write_nifti(run1_img, np.ones((2, 2, 2)))
    _write_nifti(run2_img, np.ones((2, 2, 2)))
    common_metadata = {
        'TimeZero': '12:00:00',
        'InjectionStart': 0.0,
        'FrameDuration': [10.0],
        'ImageDecayCorrectionTime': 0.0,
    }
    _write_metadata(
        run1_img.with_suffix('').with_suffix('.json'),
        {**common_metadata, 'FrameTimesStart': [0.0], 'ImageDecayCorrected': True},
    )
    _write_metadata(
        run2_img.with_suffix('').with_suffix('.json'),
        {**common_metadata, 'FrameTimesStart': [10.0], 'ImageDecayCorrected': False},
    )
    layout = BIDSLayout(bids_dir, validate=False)
    monkeypatch.setattr('petprep.utils.bids.which', lambda _: None)

    with pytest.raises(ValueError, match='decay-corrected and uncorrected'):
        combine_pet_runs(
            bids_dir=bids_dir,
            layout=layout,
            work_dir=tmp_path / 'work',
            subjects=['01'],
            bids_filters={},
        )
