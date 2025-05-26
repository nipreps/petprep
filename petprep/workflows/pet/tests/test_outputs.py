import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton
from pathlib import Path

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..outputs import (
    init_ds_petref_wf,
    init_ds_petmask_wf,
    init_ds_pet_native_wf,
    init_ds_volumes_wf,
)


def _prep_bids(tmp_path: Path) -> Path:
    bids_dir = tmp_path / "bids"
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    for p in bids_dir.rglob("*.nii.gz"):
        img.to_filename(p)
    return bids_dir


def test_datasink_datatype(tmp_path: Path):
    bids_dir = _prep_bids(tmp_path)
    out_dir = tmp_path / "out"
    with mock_config(bids_dir=bids_dir):
        wf = init_ds_petref_wf(bids_root=bids_dir, output_dir=out_dir, desc="hmc")
        assert wf.get_node("ds_petref").inputs.datatype == "pet"
        wf = init_ds_petmask_wf(output_dir=out_dir, desc="brain")
        assert wf.get_node("ds_petmask").inputs.datatype == "pet"
        wf = init_ds_pet_native_wf(
            bids_root=bids_dir,
            output_dir=out_dir,
            pet_output=True,
            all_metadata=[{}],
        )
        assert wf.get_node("ds_pet").inputs.datatype == "pet"
        wf = init_ds_volumes_wf(
            bids_root=bids_dir,
            output_dir=out_dir,
            metadata={},
        )
        assert wf.get_node("ds_pet").inputs.datatype == "pet"
        assert wf.get_node("ds_ref").inputs.datatype == "pet"
        assert wf.get_node("ds_mask").inputs.datatype == "pet"