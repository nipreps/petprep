from pathlib import Path
from unittest.mock import patch

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from ... import config
from ..base import init_petprep_wf
from ..tests import mock_config

BASE_LAYOUT = {
    '01': {
        'anat': [
            {'run': 1, 'suffix': 'T1w'},
            {'run': 2, 'suffix': 'T1w'},
            {'suffix': 'T2w'},
        ],
        'pet': [
            *(
                {
                    'task': 'rest',
                    'run': i,
                    'suffix': 'pet',
                    'metadata': {},
                }
                for i in range(1, 3)
            ),
        ],
    },
}


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

    for img_path in bids_dir.glob('sub-01/*/*.nii.gz'):
        img.to_filename(img_path)

    return bids_dir


def _make_params(
    pet2anat_init: str = 'auto',
    medial_surface_nan: bool = False,
    cifti_output: bool | str = False,
    run_msmsulc: bool = True,
    skull_strip_t1w: str = 'auto',
    use_syn_sdc: str | bool = False,
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
        use_syn_sdc,
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
        'use_syn_sdc',
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
        # Currently disabled
        # _make_params(pet2anat_init="header", force=['no-bbr']),
        _make_params(medial_surface_nan=True),
        _make_params(cifti_output='91k'),
        _make_params(cifti_output='91k', run_msmsulc=False),
        _make_params(skull_strip_t1w='force'),
        _make_params(skull_strip_t1w='skip'),
        _make_params(freesurfer=False),
        _make_params(freesurfer=False, force=['bbr']),
        _make_params(freesurfer=False, force=['no-bbr']),
        # Currently unsupported:
        # _make_params(freesurfer=False, pet2anat_init="header"),
        # _make_params(freesurfer=False, pet2anat_init="header", force=['bbr']),
        # _make_params(freesurfer=False, pet2anat_init="header", force=['no-bbr']),
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
    use_syn_sdc: str | bool,
    freesurfer: bool,
    ignore: list[str],
    force: list[str],
    bids_filters: dict,
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
        with patch.dict('fmriprep.config.execution.bids_filters', bids_filters):
            wf = init_petprep_wf()

    generate_expanded_graph(wf._create_flat_graph())
