import nibabel as nb
import numpy as np
import pytest

from ..hmc import (
    _find_highest_uptake_frame,
    get_start_frame,
    init_pet_hmc_wf,
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
