from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..fit import init_pet_fit_wf, init_pet_native_wf


@pytest.fixture(scope='module', autouse=True)
def _quiet_logger():
    import logging

    logger = logging.getLogger('nipype.workflow')
    old_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    yield
    logger.setLevel(old_level)


@pytest.fixture(scope='module')
def bids_root(tmp_path_factory):
    base = tmp_path_factory.mktemp('petfit')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


def _make_params(
    have_petref: bool = True,
    have_hmc_xfms: bool = True,
    have_petref2anat_xfm: bool = True,
):
    return (
        have_petref,
        have_hmc_xfms,
        have_petref2anat_xfm,
    )


@pytest.mark.parametrize('task', ['rest'])
@pytest.mark.parametrize(
    (
        'have_petref',
        'have_hmc_xfms',
        'have_petref2anat_xfm',
    ),
    [
        (True, True, True),
        (False, False, False),
        _make_params(have_petref=False),
        _make_params(have_hmc_xfms=False),
        _make_params(have_petref2anat_xfm=False),
    ],
)
def test_pet_fit_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    have_petref: bool,
    have_hmc_xfms: bool,
    have_petref2anat_xfm: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_series = [
            str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')
        ]

    # The workflow will attempt to read file headers
    for path in pet_series:
        img.to_filename(path)

    dummy_nifti = str(tmp_path / 'dummy.nii')
    dummy_affine = str(tmp_path / 'dummy.txt')
    img.to_filename(dummy_nifti)
    np.savetxt(dummy_affine, np.eye(4))

    # Construct precomputed files
    precomputed = {'transforms': {}}
    if have_petref:
        precomputed['petref'] = dummy_nifti
    if have_hmc_xfms:
        precomputed['transforms']['hmc'] = dummy_affine
    if have_petref2anat_xfm:
        precomputed['transforms']['petref2anat'] = dummy_affine

    with mock_config(bids_dir=bids_root):
        if have_petref != have_hmc_xfms:
            with pytest.raises(ValueError):
                init_pet_fit_wf(
                    pet_series=pet_series,
                    precomputed=precomputed,
                    omp_nthreads=1,
                )
            return

        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed=precomputed,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


@pytest.mark.parametrize('task', ['rest'])

def test_pet_native_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_series = [
            str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')
        ]

    # The workflow will attempt to read file headers
    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_native_wf(
            pet_series=pet_series,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


def test_pet_fit_mask_connections(bids_root: Path, tmp_path: Path):
    """Ensure the PET mask is generated and connected correctly."""
    pet_series = [
        str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')
    ]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))

    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed={},
            omp_nthreads=1)

    assert 'merge_mask' in wf.list_node_names()
    assert 'ds_petmask_wf.ds_petmask' in wf.list_node_names()

    merge_mask = wf.get_node('merge_mask')
    edge = wf._graph.get_edge_data(merge_mask, wf.get_node('outputnode'))
    assert ('out', 'pet_mask') in edge['connect']

    ds_edge = wf._graph.get_edge_data(merge_mask, wf.get_node('ds_petmask_wf'))
    assert ('out', 'inputnode.petmask') in ds_edge['connect']


def test_petref_report_connections(bids_root: Path, tmp_path: Path):
    """Ensure the PET reference is passed to the reports workflow."""
    pet_series = [
        str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')
    ]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))

    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed={},
            omp_nthreads=1)

    petref_buffer = wf.get_node('petref_buffer')
    edge = wf._graph.get_edge_data(petref_buffer, wf.get_node('func_fit_reports_wf'))
    assert ('petref', 'inputnode.petref') in edge['connect']


def test_pet_fit_stage1_inclusion(bids_root: Path, tmp_path: Path):
    """Stage 1 should run only when HMC derivatives are missing."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert any(name.startswith('pet_hmc_wf') for name in wf.list_node_names())

    dummy_affine = tmp_path / 'xfm.txt'
    np.savetxt(dummy_affine, np.eye(4))
    ref_file = tmp_path / 'ref.nii'
    img.to_filename(ref_file)
    precomputed = {'petref': str(ref_file), 'transforms': {'hmc': str(dummy_affine)}}

    with mock_config(bids_dir=bids_root):
        wf2 = init_pet_fit_wf(pet_series=pet_series, precomputed=precomputed, omp_nthreads=1)

    assert not any(name.startswith('pet_hmc_wf') for name in wf2.list_node_names())


def test_pet_fit_requires_both_derivatives(bids_root: Path, tmp_path: Path):
    """Supplying only one of petref or HMC transforms should raise an error."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    ref_file = tmp_path / 'ref.nii'
    hmc_xfm = tmp_path / 'xfm.txt'
    img.to_filename(ref_file)
    np.savetxt(hmc_xfm, np.eye(4))

    # Only petref provided
    with mock_config(bids_dir=bids_root):
        with pytest.raises(ValueError):
            init_pet_fit_wf(
                pet_series=pet_series,
                precomputed={'petref': str(ref_file)},
                omp_nthreads=1,
            )

    # Only hmc transforms provided
    with mock_config(bids_dir=bids_root):
        with pytest.raises(ValueError):
            init_pet_fit_wf(
                pet_series=pet_series,
                precomputed={'transforms': {'hmc': str(hmc_xfm)}},
                omp_nthreads=1,
            )
