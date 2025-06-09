from pathlib import Path

import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf


def _prep_bids(tmp_path: Path) -> Path:
    bids_dir = tmp_path / "bids"
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    for p in bids_dir.rglob("*.nii.gz"):
        img.to_filename(p)
    pet_dir = bids_dir / "sub-01" / "pet"
    pet_dir.mkdir(parents=True, exist_ok=True)
    pet_path = pet_dir / "sub-01_task-rest_run-1_pet.nii.gz"
    img.to_filename(pet_path)
    (pet_dir / "sub-01_task-rest_run-1_pet.json").write_text("{}")
    return bids_dir


def test_gtmpvc_wf(tmp_path: Path):
    bids_dir = _prep_bids(tmp_path)
    pet_series = [
        str(bids_dir / "sub-01" / "pet" / "sub-01_task-rest_run-1_pet.nii.gz")
    ]
    with mock_config(bids_dir=bids_dir):
        config.workflow.seg = "gtm"
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    pvc_wf = wf.get_node("gtmpvc_wf")
    assert pvc_wf is not None
    from nipype.interfaces.freesurfer.petsurfer import GTMPVC

    pvc = pvc_wf.get_node("gtmpvc")
    assert isinstance(pvc.interface, GTMPVC)