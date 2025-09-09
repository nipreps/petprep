from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
import yaml
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config, data
from ....utils import bids
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..fit import init_pet_fit_wf, init_pet_native_wf
from ..outputs import init_refmask_report_wf


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
    base = tmp_path_factory.mktemp('petfit')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)
    return bids_dir


def _make_params(
    have_petref: bool = True,
    have_hmc_xfms: bool = True,
    have_petref2anat_xfm: bool = True,
):
    return (
        have_petref,
        have_hmc_xfms,
        have_petref2anat_xfm,
    )


@pytest.mark.parametrize('task', ['rest'])
@pytest.mark.parametrize(
    (
        'have_petref',
        'have_hmc_xfms',
        'have_petref2anat_xfm',
    ),
    [
        (True, True, True),
        (False, False, False),
        _make_params(have_petref=False),
        _make_params(have_hmc_xfms=False),
        _make_params(have_petref2anat_xfm=False),
    ],
)
def test_pet_fit_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
    have_petref: bool,
    have_hmc_xfms: bool,
    have_petref2anat_xfm: bool,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]

    # The workflow will attempt to read file headers
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    dummy_nifti = str(tmp_path / 'dummy.nii')
    dummy_affine = str(tmp_path / 'dummy.txt')
    img.to_filename(dummy_nifti)
    np.savetxt(dummy_affine, np.eye(4))

    # Construct precomputed files
    precomputed = {'transforms': {}}
    if have_petref:
        precomputed['petref'] = dummy_nifti
    if have_hmc_xfms:
        precomputed['transforms']['hmc'] = dummy_affine
    if have_petref2anat_xfm:
        precomputed['transforms']['petref2anat'] = dummy_affine

    with mock_config(bids_dir=bids_root):
        if have_petref != have_hmc_xfms:
            with pytest.raises(ValueError):  # noqa: PT011
                init_pet_fit_wf(
                    pet_series=pet_series,
                    precomputed=precomputed,
                    omp_nthreads=1,
                )
            return

        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed=precomputed,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


@pytest.mark.parametrize('task', ['rest'])
def test_pet_native_precomputes(
    bids_root: Path,
    tmp_path: Path,
    task: str,
):
    """Test as many combinations of precomputed files and input
    configurations as possible."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    if task == 'rest':
        pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]

    # The workflow will attempt to read file headers
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        wf = init_pet_native_wf(
            pet_series=pet_series,
            omp_nthreads=1,
        )

    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)


def test_pet_fit_mask_connections(bids_root: Path, tmp_path: Path):
    """Ensure the PET mask is generated and connected correctly."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))

    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert 'merge_mask' in wf.list_node_names()
    assert 'ds_petmask_wf.ds_petmask' in wf.list_node_names()

    merge_mask = wf.get_node('merge_mask')
    edge = wf._graph.get_edge_data(merge_mask, wf.get_node('outputnode'))
    assert ('out', 'pet_mask') in edge['connect']

    ds_edge = wf._graph.get_edge_data(merge_mask, wf.get_node('ds_petmask_wf'))
    assert ('out', 'inputnode.petmask') in ds_edge['connect']


def test_petref_report_connections(bids_root: Path, tmp_path: Path):
    """Ensure the PET reference is passed to the reports workflow."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))

    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    petref_buffer = wf.get_node('petref_buffer')
    edge = wf._graph.get_edge_data(petref_buffer, wf.get_node('func_fit_reports_wf'))
    assert ('petref', 'inputnode.petref') in edge['connect']


@pytest.mark.parametrize('pvc_method', [None, 'gtm'])
def test_refmask_report_connections(bids_root: Path, tmp_path: Path, pvc_method):
    """Ensure the reference mask report is passed to the reports workflow."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    dummy_ref = str(tmp_path / 'dummy.nii')
    dummy_xfm = str(tmp_path / 'dummy.txt')
    img.to_filename(dummy_ref)
    np.savetxt(dummy_xfm, np.eye(4))
    precomputed = {
        'petref': dummy_ref,
        'transforms': {'hmc': dummy_xfm, 'petref2anat': dummy_xfm},
    }

    with mock_config(bids_dir=bids_root):
        config.workflow.ref_mask_name = 'cerebellum'
        if pvc_method is not None:
            config.workflow.pvc_method = pvc_method
        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed=precomputed,
            omp_nthreads=1,
        )

    assert 'ds_refmask_wf.ds_refmask' in wf.list_node_names()
    ref_ds = wf.get_node('ds_refmask_wf').get_node('ds_refmask')
    assert ref_ds.inputs.desc == 'refmask'
    assert ref_ds.inputs.ref == 'cerebellum'
    assert 'func_fit_reports_wf.pet_t1_refmask_report' in wf.list_node_names()
    reports_node = wf.get_node('func_fit_reports_wf')
    edge = wf._graph.get_edge_data(wf.get_node('outputnode'), reports_node)
    assert ('refmask', 'inputnode.refmask') in edge['connect']

    petref_buffer_node = wf.get_node('petref_buffer')
    ds_refmask = wf.get_node('ds_refmask_wf')
    petref_edge = wf._graph.get_edge_data(petref_buffer_node, ds_refmask)
    assert (
        'pet_file',
        'inputnode.source_files',
    ) in petref_edge['connect']

    gm_node = wf.get_node('select_gm_probseg')
    edge_prob = wf._graph.get_edge_data(gm_node, wf.get_node('pet_refmask_wf'))
    assert ('out', 'inputnode.gm_probseg') in edge_prob['connect']

    assert any(name.startswith('pet_ref_tacs_wf') for name in wf.list_node_names())
    if pvc_method is None:
        assert 'ds_ref_tacs' in wf.list_node_names()
        ds_tacs = wf.get_node('ds_ref_tacs')
        assert ds_tacs.inputs.ref == 'cerebellum'
        assert ds_tacs.inputs.seg == config.workflow.seg
        assert ds_tacs.inputs.desc == 'preproc'
        edge_tacs = wf._graph.get_edge_data(wf.get_node('pet_ref_tacs_wf'), ds_tacs)
        assert ('outputnode.timeseries', 'in_file') in edge_tacs['connect']
    else:
        assert 'ds_ref_tacs' not in wf.list_node_names()


def test_seg_entity_atlas(bids_root: Path, tmp_path: Path):
    """Reference outputs use atlas metadata when configured."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    dummy_ref = str(tmp_path / 'dummy.nii')
    dummy_xfm = str(tmp_path / 'dummy.txt')
    img.to_filename(dummy_ref)
    np.savetxt(dummy_xfm, np.eye(4))
    precomputed = {
        'petref': dummy_ref,
        'transforms': {'hmc': dummy_xfm, 'petref2anat': dummy_xfm},
    }

    with mock_config(bids_dir=bids_root):
        config.workflow.atlas = 'foo'
        config.workflow.ref_mask_name = 'cerebellum'
        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed=precomputed,
            omp_nthreads=1,
        )

    extract = wf.get_node('pet_refmask_wf').get_node('extract_refregion')
    assert extract.inputs.segmentation_type == config.workflow.atlas
    ds_tacs = wf.get_node('ds_ref_tacs')
    assert ds_tacs.inputs.seg == config.workflow.atlas
    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('pet_refmask_wf'))
    assert ('segmentation', 'inputnode.seg_file') in edge['connect']


def test_pet_fit_stage1_inclusion(bids_root: Path, tmp_path: Path):
    """Stage 1 should run only when HMC derivatives are missing."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert any(name.startswith('pet_hmc_wf') for name in wf.list_node_names())

    dummy_affine = tmp_path / 'xfm.txt'
    np.savetxt(dummy_affine, np.eye(4))
    ref_file = tmp_path / 'ref.nii'
    img.to_filename(ref_file)
    precomputed = {'petref': str(ref_file), 'transforms': {'hmc': str(dummy_affine)}}

    with mock_config(bids_dir=bids_root):
        wf2 = init_pet_fit_wf(pet_series=pet_series, precomputed=precomputed, omp_nthreads=1)

    assert not any(name.startswith('pet_hmc_wf') for name in wf2.list_node_names())


def test_pet_fit_requires_both_derivatives(bids_root: Path, tmp_path: Path):
    """Supplying only one of petref or HMC transforms should raise an error."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    ref_file = tmp_path / 'ref.nii'
    hmc_xfm = tmp_path / 'xfm.txt'
    img.to_filename(ref_file)
    np.savetxt(hmc_xfm, np.eye(4))

    # Only petref provided
    with mock_config(bids_dir=bids_root):
        with pytest.raises(ValueError):  # noqa: PT011
            init_pet_fit_wf(
                pet_series=pet_series,
                precomputed={'petref': str(ref_file)},
                omp_nthreads=1,
            )

    # Only hmc transforms provided
    with mock_config(bids_dir=bids_root):
        with pytest.raises(ValueError):  # noqa: PT011
            init_pet_fit_wf(
                pet_series=pet_series,
                precomputed={'transforms': {'hmc': str(hmc_xfm)}},
                omp_nthreads=1,
            )


def test_pet_fit_stage1_with_cached_baseline(bids_root: Path, tmp_path: Path):
    """Providing only HMC-named derivatives should skip Stage 1."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    deriv_root = tmp_path / 'derivs'
    petref = deriv_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_desc-hmc_petref.nii.gz'
    xfm = (
        deriv_root
        / 'sub-01'
        / 'pet'
        / 'sub-01_task-rest_run-1_from-orig_to-petref_mode-image_xfm.txt'
    )
    petref.parent.mkdir(parents=True)
    img.to_filename(petref)
    np.savetxt(xfm, np.eye(4))

    # ensure required metadata is present
    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    entities = bids.extract_entities(pet_series)
    precomputed = bids.collect_derivatives(derivatives_dir=deriv_root, entities=entities)

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed=precomputed, omp_nthreads=1)

    assert not any(name.startswith('pet_hmc_wf') for name in wf.list_node_names())


def test_init_refmask_report_wf(tmp_path: Path):
    """Ensure the refmask report workflow initializes without errors."""
    wf = init_refmask_report_wf(output_dir=str(tmp_path), ref_name='test')
    assert 'mask_report' in wf.list_node_names()
    ds = wf.get_node('ds_report_refmask')
    assert ds.inputs.desc == 'refmask'
    assert ds.inputs.ref == 'test'
    assert ds.inputs.suffix == 'pet'


def test_reports_spec_contains_refmask():
    """Check that the report specification includes the refmask reportlet."""
    for fname in ('reports-spec.yml', 'reports-spec-pet.yml'):
        spec = yaml.safe_load((data.load.readable(fname)).read_text())
        pet_section = next(s for s in spec['sections'] if s['name'] == 'PET')
        assert any(r.get('bids', {}).get('desc') == 'refmask' for r in pet_section['reportlets'])


def test_refmask_reports_omitted(bids_root: Path, tmp_path: Path):
    """Ensure reference mask reportlets are omitted when no reference mask is configured."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert 'func_fit_reports_wf.ds_report_refmask' not in wf.list_node_names()


def test_crop_nodes_present(bids_root: Path, tmp_path: Path):
    """Ensure crop nodes are included in the reporting workflow."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    reports = wf.get_node('func_fit_reports_wf')
    assert 'crop_petref' in reports.list_node_names()
    assert 'crop_t1w_petref' in reports.list_node_names()
    assert 'crop_petref_wm' in reports.list_node_names()
