from pathlib import Path

import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton

from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from ....interfaces import DerivativesDataSink


def _prep_bids(tmp_path: Path) -> Path:
    bids_dir = tmp_path / "bids"
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    for p in bids_dir.rglob("*.nii.gz"):
        img.to_filename(p)
    pet_dir = bids_dir / "sub-01" / "pet"
    pet_dir.mkdir(parents=True, exist_ok=True)
    img.to_filename(pet_dir / "sub-01_task-rest_run-1_pet.nii.gz")
    (pet_dir / "sub-01_task-rest_run-1_pet.json").write_text("{}")
    return bids_dir


def test_std_space_seg(tmp_path: Path):
    bids_dir = _prep_bids(tmp_path)
    pet_series = [
        str(bids_dir / "sub-01" / "pet" / "sub-01_task-rest_run-1_pet.nii.gz")
    ]
    with mock_config(bids_dir=bids_dir):
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    seg_std = wf.get_node("seg_std")
    assert isinstance(seg_std.interface, ApplyTransforms)
    assert seg_std.interface.inputs.interpolation == "MultiLabel"

    ds_wf = wf.get_node("ds_seg_std_wf")
    ds = ds_wf.get_node("ds_seg")
    assert isinstance(ds.interface, DerivativesDataSink)

    edge = wf._graph.get_edge_data(seg_std, ds_wf)
    assert ("output_image", "inputnode.segmentation") in edge["connect"]