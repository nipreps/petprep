from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf


@pytest.fixture(scope='module')
def bids_root(tmp_path_factory):
    base = tmp_path_factory.mktemp('petseg')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


@pytest.mark.parametrize(
    'seg',
    ['gtm', 'brainstem', 'thalamicNuclei', 'hippocampusAmygdala', 'wm', 'raphe', 'limbic'],
)
def test_segmentation_branch(bids_root: Path, tmp_path: Path, seg: str):
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 10)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        config.workflow.seg = seg
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    assert wf.get_node(f'pet_{seg}_seg_wf') is not None