from pathlib import Path

import nibabel as nb
import numpy as np
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf


def _prep_bids(tmp_path: Path) -> Path:
    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    for p in bids_dir.rglob('*.nii.gz'):
        img.to_filename(p)
    return bids_dir


def test_gtm_tacs_wf(tmp_path: Path):
    bids_dir = _prep_bids(tmp_path)
    pet_series = [str(bids_dir / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    with mock_config(bids_dir=bids_dir):
        config.workflow.seg = 'gtm'
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    tacs_wf = wf.get_node('gtm_tacs_wf')
    assert tacs_wf is not None

    ds = tacs_wf.get_node('ds_gtmtacs')
    from ....interfaces import DerivativesDataSink

    assert isinstance(ds.interface, DerivativesDataSink)
    assert ds.interface.inputs.datatype == 'pet'

    resample = tacs_wf.get_node('resample_seg')
    from nipype.interfaces.freesurfer import MRIConvert

    assert isinstance(resample.interface, MRIConvert)
    edge = tacs_wf._graph.get_edge_data(tacs_wf.get_node('inputnode'), resample)
    assert ('segmentation', 'in_file') in edge['connect']
    assert ('pet', 'reslice_like') in edge['connect']
    edge = tacs_wf._graph.get_edge_data(resample, tacs_wf.get_node('extract_tacs'))
    assert ('out_file', 'segmentation') in edge['connect']
