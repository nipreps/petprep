from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..base import init_pet_wf


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
    base = tmp_path_factory.mktemp('petbase')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


@pytest.mark.parametrize('task', ['rest'])
@pytest.mark.parametrize('level', ['minimal', 'resampling', 'full'])
@pytest.mark.parametrize('pet2anat_init', ['t1w', 't2w'])
@pytest.mark.parametrize('freesurfer', [False, True])
def test_pet_wf(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    level: str,
    pet2anat_init: str,
    freesurfer: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_series = [
            str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz'),
        ]

    # The workflow will attempt to read file headers
    for path in pet_series:
        img.to_filename(path)
        sidecar = Path(path).with_suffix('').with_suffix('.json')
        sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    # Toggle running recon-all
    freesurfer = bool(freesurfer)

    with mock_config(bids_dir=bids_root):
        config.workflow.pet2anat_init = pet2anat_init
        config.workflow.level = level
        config.workflow.run_reconall = freesurfer
        wf = init_pet_wf(
            pet_series=pet_series,
            precomputed={},
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


def _prep_pet_series(bids_root: Path) -> list[str]:
    """Generate dummy PET data for testing."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        sidecar = Path(path).with_suffix('').with_suffix('.json')
        sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')
    return pet_series


def test_pet_wf_with_pvc(bids_root: Path):
    """PET workflow includes the PVC workflow when configured."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.workflow.pvc_tool = 'PETPVC'
        config.workflow.pvc_method = 'GTM'
        config.workflow.pvc_psf = (1.0, 1.0, 1.0)
        config.workflow.ref_mask_name = 'cerebellum'

        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    assert any(n.startswith('pet_pvc_wf') for n in wf.list_node_names())


def test_pet_wf_without_pvc(bids_root: Path):
    """PET workflow does not include the PVC workflow by default."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.workflow.pvc_tool = None
        config.workflow.pvc_method = None
        config.workflow.pvc_psf = None
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    assert not any(n.startswith('pet_pvc_wf') for n in wf.list_node_names())


def test_pvc_entity_added(bids_root: Path):
    """Outputs include the ``pvc`` entity when PVC is run."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.execution.output_spaces = 'T1w'
        config.init_spaces()
        config.workflow.pvc_tool = 'PETPVC'
        config.workflow.pvc_method = 'GTM'
        config.workflow.pvc_psf = (1.0, 1.0, 1.0)
        config.workflow.ref_mask_name = 'cerebellum'

        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    pvc_method = config.workflow.pvc_method
    assert wf.get_node('ds_pet_t1_wf.ds_pet').inputs.pvc == pvc_method

    if 'ds_pet_std_wf.ds_pet' not in wf.list_node_names():
        pytest.skip('Standard-space datasink not created - template data may be missing.')
    assert wf.get_node('ds_pet_std_wf.ds_pet').inputs.pvc == pvc_method

    if 'pet_surf_wf.ds_pet_surfs' in wf.list_node_names():
        assert wf.get_node('pet_surf_wf.ds_pet_surfs').inputs.pvc == pvc_method

    if 'ds_pet_cifti' in wf.list_node_names():
        assert wf.get_node('ds_pet_cifti').inputs.pvc == pvc_method

    assert wf.get_node('ds_pet_tacs').inputs.pvc == pvc_method
    pet_tacs_meta = wf.get_node('ds_pet_tacs').inputs.meta_dict
    assert pet_tacs_meta['PVCMethod'] == pvc_method
    assert pet_tacs_meta['FWHM_x'] == 1.0
    assert pet_tacs_meta['FWHM_y'] == 1.0
    assert pet_tacs_meta['FWHM_z'] == 1.0
    assert pet_tacs_meta['SoftwareName'] == 'petpvc'
    assert 'CommandLine' in pet_tacs_meta

    if 'ds_ref_tacs' in wf.list_node_names():
        assert wf.get_node('ds_ref_tacs').inputs.pvc == pvc_method
        ref_tacs_meta = wf.get_node('ds_ref_tacs').inputs.meta_dict
        assert ref_tacs_meta['PVCMethod'] == pvc_method
        assert ref_tacs_meta['FWHM_x'] == 1.0
        assert ref_tacs_meta['FWHM_y'] == 1.0
        assert ref_tacs_meta['FWHM_z'] == 1.0
        assert ref_tacs_meta['SoftwareName'] == 'petpvc'
        assert 'CommandLine' in ref_tacs_meta


def test_pvc_used_in_std_space(bids_root: Path):
    """Standard-space outputs should originate from PVC data when enabled."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.workflow.pvc_tool = 'PETPVC'
        config.workflow.pvc_method = 'GTM'
        config.workflow.pvc_psf = (1.0, 1.0, 1.0)

        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    if 'pet_std_wf' not in wf.list_node_names():
        pytest.skip('Standard-space workflow not created - template data may be missing.')

    # Connection from PVC workflow to standard-space workflow
    edge = wf._graph.get_edge_data(wf.get_node('pet_pvc_wf'), wf.get_node('pet_std_wf'))
    assert ('outputnode.pet_pvc_file', 'inputnode.pet_file') in edge['connect']

    # Ensure uncorrected PET is not used as the source image
    edge_native = wf._graph.get_edge_data(wf.get_node('pet_native_wf'), wf.get_node('pet_std_wf'))
    assert ('outputnode.pet_minimal', 'inputnode.pet_file') not in edge_native['connect']
    assert ('outputnode.motion_xfm', 'inputnode.motion_xfm') not in edge_native['connect']

    edge_fit = wf._graph.get_edge_data(wf.get_node('pet_fit_wf'), wf.get_node('pet_std_wf'))
    assert ('outputnode.petref2anat_xfm', 'inputnode.petref2anat_xfm') not in edge_fit['connect']


def test_std_space_connections_without_pvc(bids_root: Path):
    """Standard-space workflow should use native outputs when PVC is disabled."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    if 'pet_std_wf' not in wf.list_node_names():
        pytest.skip('Standard-space workflow not created - template data may be missing.')

    edge = wf._graph.get_edge_data(wf.get_node('pet_native_wf'), wf.get_node('pet_std_wf'))
    assert ('outputnode.pet_minimal', 'inputnode.pet_file') in edge['connect']
    assert ('outputnode.motion_xfm', 'inputnode.motion_xfm') in edge['connect']

    edge_fit = wf._graph.get_edge_data(wf.get_node('pet_fit_wf'), wf.get_node('pet_std_wf'))
    assert ('outputnode.petref2anat_xfm', 'inputnode.petref2anat_xfm') in edge_fit['connect']


def test_pvc_receives_segmentation(bids_root: Path):
    """PVC workflow should receive segmentation from the fit workflow."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.workflow.pvc_tool = 'PETPVC'
        config.workflow.pvc_method = 'GTM'
        config.workflow.pvc_psf = (1.0, 1.0, 1.0)

        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('pet_pvc_wf'))
    assert ('segmentation', 'inputnode.segmentation') in edge['connect']


def test_pet_tacs_wf_connections(bids_root: Path):
    """TACs workflow connects expected inputs and outputs."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.execution.output_spaces = 'T1w'
        config.init_spaces()
        config.workflow.pvc_tool = None
        config.workflow.pvc_method = None
        config.workflow.pvc_psf = None
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    assert any(n.startswith('pet_tacs_wf') for n in wf.list_node_names())

    edge_anat = wf._graph.get_edge_data(wf.get_node('pet_anat_wf'), wf.get_node('pet_tacs_wf'))
    assert ('outputnode.pet_file', 'inputnode.pet_anat') in edge_anat['connect']

    edge_input = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('pet_tacs_wf'))
    assert ('segmentation', 'inputnode.segmentation') in edge_input['connect']
    assert ('dseg_tsv', 'inputnode.dseg_tsv') in edge_input['connect']

    edge_ds = wf._graph.get_edge_data(wf.get_node('pet_tacs_wf'), wf.get_node('ds_pet_tacs'))
    assert ('outputnode.timeseries', 'in_file') in edge_ds['connect']


def test_pet_ref_tacs_wf_connections(bids_root: Path):
    """Reference TAC workflow connects expected inputs and outputs."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.workflow.ref_mask_name = 'cerebellum'
        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    assert any(n.startswith('pet_ref_tacs_wf') for n in wf.list_node_names())

    edge_anat = wf._graph.get_edge_data(wf.get_node('pet_anat_wf'), wf.get_node('pet_ref_tacs_wf'))
    assert ('outputnode.pet_file', 'inputnode.pet_anat') in edge_anat['connect']

    edge_fit = wf._graph.get_edge_data(wf.get_node('pet_fit_wf'), wf.get_node('pet_ref_tacs_wf'))
    assert ('outputnode.refmask', 'inputnode.mask_file') in edge_fit['connect']

    edge_ds = wf._graph.get_edge_data(wf.get_node('pet_ref_tacs_wf'), wf.get_node('ds_ref_tacs'))
    assert ('outputnode.timeseries', 'in_file') in edge_ds['connect']


def test_psf_metadata_propagation(bids_root: Path):
    """PSF values should be passed to datasinks when using AGTM."""
    pet_series = _prep_pet_series(bids_root)

    with mock_config(bids_dir=bids_root):
        config.execution.output_spaces = 'T1w'
        config.init_spaces()
        config.workflow.pvc_tool = 'petsurfer'
        config.workflow.pvc_method = 'AGTM'
        config.workflow.pvc_psf = (1.0,)

        wf = init_pet_wf(pet_series=pet_series, precomputed={})

    edge = wf._graph.get_edge_data(wf.get_node('pet_pvc_wf'), wf.get_node('ds_pet_t1_wf.psf_meta'))
    assert ('outputnode.fwhm_x', 'fwhm_x') in edge['connect']

    edge_ds = wf._graph.get_edge_data(
        wf.get_node('ds_pet_t1_wf.psf_meta'), wf.get_node('ds_pet_t1_wf.ds_pet')
    )
    assert ('meta_dict', 'meta_dict') in edge_ds['connect']
