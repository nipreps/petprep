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

    seg_wf = wf.get_node(f'pet_{seg}_seg_wf')
    assert seg_wf is not None

    if seg == 'gtm':
        run_gtm = seg_wf.get_node('run_gtm')
        from nipype.interfaces.freesurfer.petsurfer import GTMSeg
        from nipype.interfaces.freesurfer import MRIConvert
        from ....interfaces import DerivativesDataSink

        assert isinstance(run_gtm.interface, GTMSeg)
        assert hasattr(run_gtm.inputs, 'subject_id')
        assert hasattr(run_gtm.inputs, 'subjects_dir')

        convert = seg_wf.get_node('convert_gtmseg')
        ds = seg_wf.get_node('ds_gtmseg')
        dseg_tsv = seg_wf.get_node('ds_gtmdsegtsv')
        morph_tsv = seg_wf.get_node('ds_gtmmorphtsv')

        assert isinstance(convert.interface, MRIConvert)
        assert convert.interface.out_type == 'niigz'
        assert isinstance(ds.interface, DerivativesDataSink)
        assert ds.interface.inputs.desc == 'gtm'
        assert ds.interface.inputs.suffix == 'dseg'
        assert ds.interface.inputs.compress is True
        assert isinstance(dseg_tsv.interface, DerivativesDataSink)
        assert dseg_tsv.interface.inputs.desc == 'gtm'
        assert dseg_tsv.interface.inputs.suffix == 'dseg'
        assert dseg_tsv.interface.inputs.extension == '.tsv'
        assert isinstance(morph_tsv.interface, DerivativesDataSink)
        assert morph_tsv.interface.inputs.desc == 'gtm'
        assert morph_tsv.interface.inputs.suffix == 'morph'
        assert morph_tsv.interface.inputs.extension == '.tsv'
        edge = seg_wf._graph.get_edge_data(seg_wf.get_node('inputnode'), convert)
        assert ('t1w_preproc', 'reslice_like') in edge['connect']
