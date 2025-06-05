from pathlib import Path
from unittest.mock import patch
import json

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from ... import config
from ..base import init_petprep_wf
from ..tests import mock_config

from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES
from niworkflows.utils.bids import collect_data as original_collect_data
import copy

BASE_LAYOUT = {
    '01': {
        'anat': [
            {'suffix': 'T1w'},
            {'suffix': 'inplaneT2'},
        ],
        'pet': [
            {
                'suffix': 'pet',
                'metadata': {},
            },
        ],
        'func': [
            {'task': 'mixedgamblestask', 'run': 1, 'suffix': 'bold'},
            {'task': 'mixedgamblestask', 'run': 2, 'suffix': 'bold'},
            {'task': 'mixedgamblestask', 'run': 3, 'suffix': 'bold'},
        ],
    },
}


@pytest.fixture(scope='module')
def custom_queries():
    queries = copy.deepcopy(DEFAULT_BIDS_QUERIES)
    queries['pet'] = {'datatype': 'pet', 'suffix': 'pet'}
    queries['t1w'].pop('datatype', None)
    return queries


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
    base = tmp_path_factory.mktemp('base')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    # anat files
    anat_dir = bids_dir / "sub-01" / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    img.to_filename(anat_dir / "sub-01_T1w.nii.gz")
    img.to_filename(anat_dir / "sub-01_inplaneT2.nii.gz")

    # pet file
    pet_dir = bids_dir / "sub-01" / "pet"
    pet_dir.mkdir(parents=True, exist_ok=True)
    pet_path = pet_dir / "sub-01_pet.nii.gz"
    img.to_filename(pet_path)
    
    # Add metadata explicitly
    metadata = {}
    json_path = pet_dir / "sub-01_pet.json"
    json_path.write_text(json.dumps(metadata))

    # func files (optional for PET workflow but included for consistency)
    func_dir = bids_dir / "sub-01" / "func"
    func_dir.mkdir(parents=True, exist_ok=True)
    for run in range(1, 4):
        func_path = func_dir / f"sub-01_task-mixedgamblestask_run-0{run}_bold.nii.gz"
        img.to_filename(func_path)
        events_path = func_dir / f"sub-01_task-mixedgamblestask_run-0{run}_events.tsv"
        events_path.write_text("onset\tduration\ttrial_type\n")

    return bids_dir


def _make_params(
    pet2anat_init: str = 'auto',
    medial_surface_nan: bool = False,
    cifti_output: bool | str = False,
    run_msmsulc: bool = True,
    skull_strip_t1w: str = 'auto',
    freesurfer: bool = True,
    ignore: list[str] = None,
    force: list[str] = None,
    bids_filters: dict = None,
):
    if ignore is None:
        ignore = []
    if force is None:
        force = []
    if bids_filters is None:
        bids_filters = {}
    return (
        pet2anat_init,
        medial_surface_nan,
        cifti_output,
        run_msmsulc,
        skull_strip_t1w,
        freesurfer,
        ignore,
        force,
        bids_filters,
    )


@pytest.mark.parametrize('level', ['minimal', 'resampling', 'full'])
@pytest.mark.parametrize('anat_only', [False, True])
@pytest.mark.parametrize(
    (
        'pet2anat_init',
        'medial_surface_nan',
        'cifti_output',
        'run_msmsulc',
        'skull_strip_t1w',
        'freesurfer',
        'ignore',
        'force',
        'bids_filters',
    ),
    [
        _make_params(),
        _make_params(pet2anat_init='t1w'),
        _make_params(pet2anat_init='t2w'),
        _make_params(pet2anat_init='header'),
        _make_params(force=['bbr']),
        _make_params(force=['no-bbr']),
        _make_params(pet2anat_init='header', force=['bbr']),
        _make_params(medial_surface_nan=True),
        _make_params(cifti_output='91k'),
        _make_params(cifti_output='91k', run_msmsulc=False),
        _make_params(skull_strip_t1w='force'),
        _make_params(skull_strip_t1w='skip'),
        _make_params(freesurfer=False),
        _make_params(freesurfer=False, force=['bbr']),
        _make_params(freesurfer=False, force=['no-bbr']),
    ],
)
def test_init_petprep_wf(
    bids_root: Path,
    tmp_path: Path,
    level: str,
    anat_only: bool,
    pet2anat_init: str,
    medial_surface_nan: bool,
    cifti_output: bool | str,
    run_msmsulc: bool,
    skull_strip_t1w: str,
    freesurfer: bool,
    ignore: list[str],
    force: list[str],
    bids_filters: dict,
    custom_queries: dict,
):
    with mock_config(bids_dir=bids_root):
        config.workflow.level = level
        config.workflow.anat_only = anat_only
        config.workflow.pet2anat_init = pet2anat_init
        config.workflow.medial_surface_nan = medial_surface_nan
        config.workflow.run_msmsulc = run_msmsulc
        config.workflow.skull_strip_t1w = skull_strip_t1w
        config.workflow.cifti_output = cifti_output
        config.workflow.run_reconall = freesurfer
        config.workflow.ignore = ignore
        config.workflow.force = force

        with patch.dict('petprep.config.execution.bids_filters', bids_filters):
            # Patch the correct function with the correct return value explicitly
            with patch('niworkflows.utils.bids.collect_data') as mock_collect_data:
                mock_collect_data.return_value = original_collect_data(
                    bids_root,
                    '01',
                    bids_filters=bids_filters,
                    queries=custom_queries,
                )

                wf = init_petprep_wf()

    generate_expanded_graph(wf._create_flat_graph())