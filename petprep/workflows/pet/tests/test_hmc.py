import nibabel as nb
import numpy as np
import pytest

from ..hmc import (
    _find_highest_uptake_frame,
    estimate_hmc_mem_usage,
    get_start_frame,
    init_pet_hmc_wf,
    plan_hmc_resource_policy,
    update_list_transforms,
)


def test_get_start_frame_basic():
    durations = [60, 60, 60]
    assert get_start_frame(durations, 120) == 2
    assert get_start_frame(durations, 0) == 0
    # start time greater than all midpoints should return last index
    assert get_start_frame(durations, 200) == 2


def test_get_start_frame_with_starts():
    durations = [30, 30, 30]
    frame_starts = [0, 40, 80]
    assert get_start_frame(durations, 15, frame_starts) == 1


def test_get_start_frame_empty():
    assert get_start_frame([], 50) == 0
    assert get_start_frame(None, 50) == 0


def test_update_list_transforms_padding():
    xforms = ['a', 'b', 'c']
    assert update_list_transforms(xforms, 2) == ['a', 'a', 'a', 'b', 'c']
    assert update_list_transforms(xforms, 0) == xforms


def test_update_list_transforms_empty():
    with pytest.raises(ValueError, match='cannot be empty'):
        update_list_transforms([], 1)


def test_init_pet_hmc_wf_nodes():
    wf = init_pet_hmc_wf(mem_gb=1, omp_nthreads=1)
    names = wf.list_node_names()
    assert 'split_frames' in names
    assert 'est_robust_hmc' in names
    assert 'convert_ref' in names


def test_init_pet_hmc_wf_auto_inittp():
    wf = init_pet_hmc_wf(mem_gb=1, omp_nthreads=1, initial_frame='auto')
    names = wf.list_node_names()
    assert 'find_highest_uptake_frame' in names


def test_init_pet_hmc_wf_specific_inittp():
    wf = init_pet_hmc_wf(mem_gb=1, omp_nthreads=1, initial_frame=2, fixed_frame=True)
    names = wf.list_node_names()
    assert 'find_highest_uptake_frame' not in names
    node = wf.get_node('est_robust_hmc')
    initial_frame = 2
    assert node.inputs.initial_timepoint == initial_frame + 1
    assert node.inputs.fixed_timepoint is True
    assert node.inputs.no_iteration is True


def test_init_pet_hmc_wf_subsample_threshold():
    wf = init_pet_hmc_wf(
        mem_gb=2,
        omp_nthreads=1,
        fixed_frame=True,
        subsample_threshold=200,
    )
    node = wf.get_node('est_robust_hmc')
    assert node.inputs.subsample_threshold == 200
    assert node.mem_gb == 2


def test_find_highest_uptake_frame(tmp_path):
    data = [np.ones((2, 2, 2)) * i for i in (1, 2, 3)]
    files = []
    for idx, arr in enumerate(data):
        img = nb.Nifti1Image(arr, np.eye(4))
        fname = tmp_path / f'frame{idx}.nii.gz'
        img.to_filename(fname)
        files.append(str(fname))

    expected = np.argmax([arr.sum() for arr in data]) + 1
    result = _find_highest_uptake_frame(files)
    assert result == expected


def test_estimate_hmc_mem_usage_respects_start_time(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 5), dtype=np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    estimate = estimate_hmc_mem_usage(
        str(pet_file),
        start_time=2.0,
        frame_durations=[1, 1, 1, 1, 1],
        frame_start_times=[0, 1, 2, 3, 4],
    )

    assert estimate['start_frame'] == 2
    assert estimate['selected_frames'] == 3
    assert estimate['total_frames'] == 5
    assert estimate['estimate'] > estimate['frame']


def test_estimate_hmc_mem_usage_subsample_threshold(tmp_path):
    img = nb.Nifti1Image(np.zeros((30, 25, 22, 2), dtype=np.uint8), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    full = estimate_hmc_mem_usage(str(pet_file), subsample_threshold=None)
    subsampled = estimate_hmc_mem_usage(str(pet_file), subsample_threshold=20)

    assert subsampled['estimate'] < full['estimate']
    assert subsampled['frame'] < full['frame']


def test_plan_hmc_resource_policy_ignores_requested_memory(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 4), dtype=np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    policy = plan_hmc_resource_policy(
        str(pet_file),
        frame_durations=[1, 1, 1, 1],
        frame_start_times=[0, 1, 2, 3],
        fixed_frame=False,
    )

    assert policy['auto_limited'] is False
    assert policy['fixed_frame'] is False
    assert policy['subsample_threshold'] is None


def test_plan_hmc_resource_policy_allows_forty_lowres_frames(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 40), dtype=np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    policy = plan_hmc_resource_policy(
        str(pet_file),
        start_time=0,
        frame_durations=[1] * 40,
        frame_start_times=list(range(40)),
    )

    assert policy['selected_frames'] == 40
    assert policy['auto_limited'] is False
    assert policy['fixed_frame'] is False
    assert policy['subsample_threshold'] is None


def test_plan_hmc_resource_policy_limits_more_than_forty_frames(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 41), dtype=np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    policy = plan_hmc_resource_policy(
        str(pet_file),
        start_time=0,
        frame_durations=[1] * 41,
        frame_start_times=list(range(41)),
    )

    assert policy['selected_frames'] == 41
    assert policy['auto_limited'] is True
    assert policy['fixed_frame'] is True
    assert policy['subsample_threshold'] is None
    assert '41 selected frames' in policy['reason']


def test_plan_hmc_resource_policy_skips_subsample_below_floor(tmp_path):
    img = nb.Nifti1Image(np.zeros((100, 100, 100, 41), dtype=np.uint8), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    policy = plan_hmc_resource_policy(
        str(pet_file),
        start_time=0,
        frame_durations=[1] * 41,
        frame_start_times=list(range(41)),
    )

    assert policy['selected_frames'] == 41
    assert policy['auto_limited'] is True
    assert policy['fixed_frame'] is True
    assert policy['subsample_threshold'] is None
    assert policy['planned_memory_gb'] == policy['estimated_memory_gb']


def test_plan_hmc_resource_policy_selects_dynamic_subsample_threshold(tmp_path):
    img = nb.Nifti1Image(np.zeros((160, 160, 160, 41), dtype=np.uint8), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    policy = plan_hmc_resource_policy(
        str(pet_file),
        start_time=0,
        frame_durations=[1] * 41,
        frame_start_times=list(range(41)),
    )

    assert policy['selected_frames'] == 41
    assert policy['auto_limited'] is True
    assert policy['fixed_frame'] is True
    assert policy['subsample_threshold'] == 159
    assert policy['planned_memory_gb'] < policy['estimated_memory_gb']
