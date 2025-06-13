from ..hmc import get_start_frame, update_list_transforms, init_pet_hmc_wf


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


import pytest


def test_update_list_transforms_empty():
    with pytest.raises(ValueError):
        update_list_transforms([], 1)


def test_init_pet_hmc_wf_nodes():
    wf = init_pet_hmc_wf(mem_gb=1, omp_nthreads=1)
    names = wf.list_node_names()
    assert 'pet_hmc_wf.split_frames' in names
    assert 'pet_hmc_wf.est_robust_hmc' in names
    assert 'pet_hmc_wf.convert_ref' in names