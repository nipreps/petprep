import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton

from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf


def test_pet_mask_flow(tmp_path):
    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 10)), np.eye(4))
    pet_file = bids_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz'
    img.to_filename(pet_file)

    with mock_config(bids_dir=bids_dir):
        wf = init_pet_wf(pet_series=str(pet_file), precomputed={})

    edge = wf._graph.get_edge_data(
        wf.get_node('pet_fit_wf'), wf.get_node('pet_confounds_wf')
    )
    assert ('pet_mask', 'inputnode.pet_mask') in edge['connect']

    conf_wf = wf.get_node('pet_confounds_wf')
    conf_edge = conf_wf._graph.get_edge_data(
        conf_wf.get_node('inputnode'), conf_wf.get_node('dvars')
    )
    assert ('pet_mask', 'in_mask') in conf_edge['connect']