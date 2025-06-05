from pathlib import Path

import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton
import pytest

from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf

@pytest.fixture(scope='module')
def bids_root(tmp_path_factory):
    base = tmp_path_factory.mktemp('petfit')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


def test_pet_mask_flow(bids_root: Path, tmp_path: Path):
    pet_series = [
        str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')
    ]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 10)), np.eye(4))
    
    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_wf(
            pet_series=pet_series, 
            precomputed={}
        )

    assert wf is not None, "Workflow was not initialized."

    pet_fit_node = wf.get_node('pet_fit_wf')
    pet_confounds_node = wf.get_node('pet_confounds_wf')

    assert pet_fit_node is not None, "pet_fit_wf node missing"
    assert pet_confounds_node is not None, "pet_confounds_wf node missing"

    edge = wf._graph.get_edge_data(pet_fit_node, pet_confounds_node)
    assert edge is not None, "Edge missing between pet_fit_wf and pet_confounds_wf"
    
    # Correct assertion:
    assert ('outputnode.pet_mask', 'inputnode.pet_mask') in edge['connect']

    conf_edge = pet_confounds_node._graph.get_edge_data(
        pet_confounds_node.get_node('inputnode'), pet_confounds_node.get_node('dvars')
    )
    assert conf_edge is not None, "Confound edge is missing."
    assert ('pet_mask', 'in_mask') in conf_edge['connect']