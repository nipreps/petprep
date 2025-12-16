from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
import pytest
import yaml
from nipype.interfaces.base import Undefined
from nipype.pipeline.engine.utils import generate_expanded_graph
from niworkflows.utils.testing import generate_bids_skeleton

from .... import config, data
from ....utils import bids
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from ..fit import (
    _construct_nu_path,
    _detect_large_pet_mask,
    _extract_first5min_image,
    _extract_sum_image,
    _extract_twa_image,
    _select_anatomical_reference,
    _select_best_petref,
    _write_identity_xforms,
    init_pet_fit_wf,
    init_pet_native_wf,
)
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


def test_reports_use_motion_corrected_average(bids_root: Path, tmp_path: Path):
    """Co-registration report should show the motion corrected time-weighted average."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 2.0)), axis=-1)
    img = nb.Nifti1Image(data, np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}')

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert 'report_petref' in wf.list_node_names()
    reports_node = wf.get_node('func_fit_reports_wf')
    report_petref = wf.get_node('report_petref')
    edge = wf._graph.get_edge_data(report_petref, reports_node)
    assert ('out_file', 'inputnode.report_pet') in edge['connect']


def test_reference_extraction_helpers(tmp_path: Path):
    pet_4d = tmp_path / 'pet.nii.gz'
    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 2.0)), axis=-1)
    nb.Nifti1Image(data, np.eye(4)).to_filename(pet_4d)

    sidecar = {'FrameTimesStart': [0.0, 60.0], 'FrameDuration': [60.0, 60.0]}
    out = _extract_twa_image(
        str(pet_4d), tmp_path, sidecar['FrameTimesStart'], sidecar['FrameDuration']
    )
    assert Path(out).name.endswith('_timeavgref.nii.gz')
    img = nb.load(out)
    assert img.shape == (2, 2, 2)
    assert np.allclose(img.get_fdata(), 1.5)

    sum_out = _extract_sum_image(str(pet_4d), tmp_path)
    assert Path(sum_out).name.endswith('_sumref.nii.gz')
    sum_img = nb.load(sum_out)
    assert np.allclose(sum_img.get_fdata(), 3.0)

    first5 = _extract_first5min_image(
        str(pet_4d),
        tmp_path,
        sidecar['FrameTimesStart'],
        sidecar['FrameDuration'],
        window_sec=30.0,
    )
    assert Path(first5).name.endswith('_first5minref.nii.gz')
    first_img = nb.load(first5)
    # Only the first frame overlaps the 30s window
    assert np.allclose(first_img.get_fdata(), 1.0)

    with pytest.raises(ValueError):
        _extract_twa_image(str(pet_4d), tmp_path, None, None)
    with pytest.raises(ValueError):
        _extract_first5min_image(str(pet_4d), tmp_path, [0.0], [1.0], window_sec=-1)


def test_petref_default_twa_when_hmc_disabled(bids_root: Path, tmp_path: Path):
    """Disabling HMC should fall back to TWA references and note it in reports."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 2.0)), axis=-1)
    img = nb.Nifti1Image(data, np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}')

    with mock_config(bids_dir=bids_root):
        config.workflow.hmc_off = True
        config.workflow.petref = 'template'
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert 'twa_reference' in wf.list_node_names()

    summary = wf.get_node('summary')
    assert summary.inputs.petref_strategy == 'twa'
    assert summary.inputs.requested_petref_strategy == 'template'
    assert summary.inputs.requested_anatref == 't1w'
    assert summary.inputs.hmc_disabled is True


def test_pet_reference_utilities(tmp_path: Path):
    labels = ['template', 'twa', 'sum']
    scores = [0.5, None, 0.25]
    transforms = ['ants', 'fs', 'fs']
    inv_transforms = ['ants_inv', 'fs_inv', 'fs_inv']
    winners = ['ants', 'fs', 'fs']
    petrefs = ['tpl.nii.gz', 'twa.nii.gz', 'sum.nii.gz']
    selection = _select_best_petref(labels, scores, transforms, inv_transforms, winners, petrefs)
    assert selection[0] == 'sum'
    assert selection[1] == 0.25

    with pytest.raises(ValueError):
        _select_best_petref([], [], [], [], [], [])
    with pytest.raises(ValueError):
        _select_best_petref(['a'], [None], ['x'], ['y'], ['w'], ['z'])

    xform_file = _write_identity_xforms(2, tmp_path / 'xfms' / 'itk.txt')
    assert xform_file.exists()

    nu_path = _construct_nu_path('/subjects', 'sub-01')
    assert nu_path.endswith('sub-01/mri/nu.mgz')


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
    assert ref_ds.inputs.desc == 'ref'
    assert ref_ds.inputs.label == 'cerebellum'
    assert 'label' in ref_ds.interface._allowed_entities
    assert 'func_fit_reports_wf.pet_t1_refmask_report' in wf.list_node_names()
    reports_node = wf.get_node('func_fit_reports_wf')
    edge = wf._graph.get_edge_data(wf.get_node('outputnode'), reports_node)
    assert ('refmask', 'inputnode.refmask') in edge['connect']

    ds_refmask = wf.get_node('ds_refmask_wf')
    gm_node = wf.get_node('select_gm_probseg')
    gm_edge = wf._graph.get_edge_data(gm_node, ds_refmask)
    assert ('out', 'inputnode.source_files') in gm_edge['connect']
    seg_edge = wf._graph.get_edge_data(wf.get_node('inputnode'), ds_refmask)
    assert ('segmentation', 'inputnode.segmentation') in seg_edge['connect']

    merge_node = ds_refmask.get_node('merge_source_files')
    merge_edge = ds_refmask._graph.get_edge_data(ds_refmask.get_node('inputnode'), merge_node)
    assert (
        'segmentation',
        'in2',
    ) in merge_edge['connect']

    edge_prob = wf._graph.get_edge_data(gm_node, wf.get_node('pet_refmask_wf'))
    assert ('out', 'inputnode.gm_probseg') in edge_prob['connect']

    assert any(name.startswith('pet_ref_tacs_wf') for name in wf.list_node_names())
    if pvc_method is None:
        assert 'ds_ref_tacs' in wf.list_node_names()
        ds_tacs = wf.get_node('ds_ref_tacs')
        assert ds_tacs.inputs.label == 'cerebellum'
        assert 'label' in ds_tacs.interface._allowed_entities
        assert 'seg' not in ds_tacs.interface._allowed_entities
        assert not hasattr(ds_tacs.inputs, 'seg')
        assert ds_tacs.inputs.desc == 'preproc'
        edge_tacs = wf._graph.get_edge_data(wf.get_node('pet_ref_tacs_wf'), ds_tacs)
        assert ('outputnode.timeseries', 'in_file') in edge_tacs['connect']
    else:
        assert 'ds_ref_tacs' not in wf.list_node_names()


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


def test_pet_fit_robust_registration(bids_root: Path, tmp_path: Path):
    """Robust PET-to-anatomical registration swaps in mri_robust_register."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.pet2anat_method = 'robust'
        config.workflow.pet2anat_dof = 6
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert 'pet_reg_wf.mri_robust_register' in node_names
    assert 'pet_reg_wf.mri_coreg' not in node_names
    assert 'pet_reg_wf.ants_registration' not in node_names


def test_init_pet_fit_wf_ants_registration(bids_root: Path, tmp_path: Path):
    """Test PET fit workflow with ANTs registration."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.pet2anat_method = 'ants'
        config.workflow.pet2anat_dof = 6
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert 'pet_reg_wf.ants_registration' in node_names
    assert 'pet_reg_wf.mri_coreg' not in node_names
    assert 'pet_reg_wf.mri_robust_register' not in node_names


def test_init_pet_fit_wf_auto_registration(bids_root: Path, tmp_path: Path):
    """Auto PET-to-anatomical registration runs and scores both branches."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.pet2anat_method = 'auto'
        config.workflow.pet2anat_dof = 6
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert 'pet_reg_wf.ants_registration' in node_names
    assert 'pet_reg_wf.mri_coreg' in node_names
    assert 'pet_reg_wf.select_best' in node_names
    assert 'pet_reg_wf.score_ants' in node_names
    assert 'pet_reg_wf.score_fs' in node_names
    assert 'pet_reg_wf.warp_pet_ants' in node_names
    assert 'pet_reg_wf.warp_pet_fs' in node_names


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


def test_pet_fit_reruns_coreg_when_default_options_specified(bids_root: Path, tmp_path: Path):
    """Explicit default CLI flags should also ignore cached transforms."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    deriv_root = tmp_path / 'derivs'
    petref = deriv_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_desc-hmc_petref.nii.gz'
    hmc_xfm = (
        deriv_root
        / 'sub-01'
        / 'pet'
        / 'sub-01_task-rest_run-1_from-orig_to-petref_mode-image_xfm.txt'
    )
    petref2anat_xfm = (
        deriv_root
        / 'sub-01'
        / 'pet'
        / 'sub-01_task-rest_run-1_from-petref_to-anat_mode-image_xfm.txt'
    )

    petref.parent.mkdir(parents=True)
    img.to_filename(petref)
    np.savetxt(hmc_xfm, np.eye(4))
    np.savetxt(petref2anat_xfm, np.eye(4))

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    entities = bids.extract_entities(pet_series)
    precomputed = bids.collect_derivatives(derivatives_dir=deriv_root, entities=entities)

    with mock_config(bids_dir=bids_root):
        config.workflow.petref_specified = True
        config.workflow.pet2anat_method_specified = True
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed=precomputed, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert 'pet_reg_wf.mri_coreg' in node_names
    assert wf.get_node('outputnode').inputs.petref2anat_xfm is Undefined


def test_pet_fit_hmc_off_disables_stage1(bids_root: Path, tmp_path: Path):
    """Disabling HMC should skip Stage 1 and use identity transforms."""
    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    data = np.stack(
        (
            np.ones((2, 2, 2), dtype=np.float32),
            np.full((2, 2, 2), 3.0, dtype=np.float32),
        ),
        axis=-1,
    )
    img = nb.Nifti1Image(data, np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0, 2], "FrameDuration": [2, 4]}')

    with mock_config(bids_dir=bids_root):
        config.workflow.hmc_off = True
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

        assert not any(name.startswith('pet_hmc_wf') for name in wf.list_node_names())
        hmc_buffer = wf.get_node('hmc_buffer')
        assert str(hmc_buffer.inputs.hmc_xforms).endswith('idmat.tfm')
        hmc = nt.linear.load(hmc_buffer.inputs.hmc_xforms)
        assert hmc.matrix.shape[0] == data.shape[-1]
        assert np.allclose(hmc.matrix, np.tile(np.eye(4), (data.shape[-1], 1, 1)))
        petref_buffer = wf.get_node('petref_buffer')
        petref_name = Path(petref_buffer.inputs.petref).name
        assert petref_name.endswith('_timeavgref.nii.gz')
        assert '.nii_timeavgref' not in petref_name
        petref_img = nb.load(petref_buffer.inputs.petref)
        assert np.allclose(petref_img.get_fdata(), 14.0 / 6.0)


@pytest.mark.parametrize(
    ('frame_start_times', 'frame_durations', 'message'),
    [
        (None, [1, 1], 'Frame timing metadata are required'),
        ([0, 1], None, 'Frame timing metadata are required'),
        ([[0, 1]], [1, 1], 'must be one-dimensional'),
        ([0, 1], [1], 'the same length'),
        ([0, 1, 2], [1, 1, 1], 'match the number of frames'),
        ([0, 1], [1, -1], 'must all be positive'),
        ([1, 0], [1, 1], 'must be non-decreasing'),
    ],
)
def test_extract_twa_image_validation(
    tmp_path: Path, frame_start_times, frame_durations, message: str
):
    """Validate error handling for malformed frame timing metadata."""

    pet_img = nb.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    pet_img.to_filename(pet_file)

    with pytest.raises(ValueError, match=message):  # noqa: PT011
        _extract_twa_image(
            str(pet_file),
            tmp_path / 'out',
            frame_start_times,
            frame_durations,
        )


def test_extract_sum_image(tmp_path: Path):
    """Summed references are written out with the expected contents."""

    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 3.0)), axis=-1)
    pet_img = nb.Nifti1Image(data.astype(np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    pet_img.to_filename(pet_file)

    out_file = _extract_sum_image(str(pet_file), tmp_path / 'out')

    summed = nb.load(out_file).get_fdata()
    assert np.allclose(summed, 4.0)
    assert Path(out_file).name == 'pet_sumref.nii.gz'

    # 3D inputs should round-trip without creating a new file
    pet_3d = tmp_path / 'pet3d.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(pet_3d)
    assert _extract_sum_image(str(pet_3d), tmp_path / 'out') == str(pet_3d)


def test_extract_first5min_image(tmp_path: Path):
    """Early reference averages only the first 5 minutes of data."""

    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 3.0)), axis=-1)
    pet_img = nb.Nifti1Image(data.astype(np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    pet_img.to_filename(pet_file)

    out_file = _extract_first5min_image(
        str(pet_file),
        tmp_path / 'out',
        frame_start_times=[0, 400],
        frame_durations=[400, 200],
    )

    averaged = nb.load(out_file).get_fdata()
    expected = (1.0 * 300 + 3.0 * 0) / 300
    assert np.allclose(averaged, expected)
    assert Path(out_file).name == 'pet_first5minref.nii.gz'


def test_extract_first5min_image_fallback_first_frame(tmp_path: Path):
    """If early frames are missing, fall back to the first frame."""

    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 5.0)), axis=-1)
    pet_img = nb.Nifti1Image(data.astype(np.float32), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    pet_img.to_filename(pet_file)

    out_file = _extract_first5min_image(
        str(pet_file),
        tmp_path / 'out',
        frame_start_times=[600, 1200],
        frame_durations=[600, 600],
        fallback_to_first_frame=True,
    )

    averaged = nb.load(out_file).get_fdata()
    assert np.allclose(averaged, 1.0)
    assert Path(out_file).name == 'pet_first5minref.nii.gz'


def test_report_petref_receives_frame_metadata(bids_root: Path, tmp_path: Path):
    """Report reference node always receives timing metadata."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.petref = 'sum'
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    report_petref = wf.get_node('report_petref')
    assert report_petref.inputs.frame_start_times == [0, 1]
    assert report_petref.inputs.frame_durations == [1, 1]


def test_pet_fit_hmc_off_ignores_precomputed(bids_root: Path, tmp_path: Path):
    """Precomputed derivatives are ignored when ``--hmc-off`` is set."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    data = np.stack((np.ones((2, 2, 2)), np.full((2, 2, 2), 2.0)), axis=-1)
    img = nb.Nifti1Image(data, np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}')

    precomputed_petref = tmp_path / 'precomputed_petref.nii.gz'
    precomputed_hmc = tmp_path / 'precomputed_hmc.txt'
    img.to_filename(precomputed_petref)
    np.savetxt(precomputed_hmc, np.eye(4))

    with mock_config(bids_dir=bids_root):
        config.workflow.hmc_off = True
        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed={
                'petref': str(precomputed_petref),
                'transforms': {'hmc': str(precomputed_hmc)},
            },
            omp_nthreads=1,
        )

    petref_buffer = wf.get_node('petref_buffer')
    hmc_buffer = wf.get_node('hmc_buffer')

    assert petref_buffer.inputs.petref != str(precomputed_petref)
    assert Path(petref_buffer.inputs.petref).name.endswith('_timeavgref.nii.gz')
    assert hmc_buffer.inputs.hmc_xforms != str(precomputed_hmc)
    assert Path(hmc_buffer.inputs.hmc_xforms).name == 'idmat.tfm'


def test_write_identity_xforms_minimum(tmp_path: Path):
    """At least one identity transform should always be written."""

    xfm_file = _write_identity_xforms(0, tmp_path / 'idmat.tfm')

    xforms = nt.linear.load(xfm_file)
    matrices = np.asarray(xforms.matrix)
    if matrices.ndim == 2:
        matrices = matrices[np.newaxis, ...]

    assert matrices.shape[0] == 1
    assert np.allclose(matrices[0], np.eye(4))


def test_select_anatomical_reference_prefers_nu(tmp_path: Path):
    """Selecting ``anatref='nu'`` should return the FreeSurfer nu image when present."""

    t1 = tmp_path / 't1.nii.gz'
    nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(t1)

    nu = tmp_path / 'nu.mgz'
    nb.MGHImage(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(nu)

    selected, label = _select_anatomical_reference('nu', str(t1), str(nu), False)

    assert label == 'nu'
    assert selected == str(nu)


def test_select_anatomical_reference_fallback(tmp_path: Path):
    """When ``anatref`` is ``'auto'`` and nu.mgz is missing, keep the T1w reference."""

    t1 = tmp_path / 't1.nii.gz'
    nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(t1)

    selected, label = _select_anatomical_reference(
        'auto', str(t1), str(tmp_path / 'missing.mgz'), True
    )

    assert label == 't1w'
    assert selected == str(t1)


def test_detect_large_pet_mask(tmp_path: Path):
    """PET masks substantially larger than the anatomical mask trigger a recommendation."""

    pet_mask = tmp_path / 'pet_mask.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)).to_filename(pet_mask)

    t1_mask = tmp_path / 't1_mask.nii.gz'
    nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)).to_filename(t1_mask)

    use_nu, ratio, pet_vol, t1_vol = _detect_large_pet_mask(str(pet_mask), str(t1_mask))

    assert use_nu is True
    assert ratio > 1.5
    assert pet_vol > t1_vol


def test_detect_large_pet_mask_within_threshold(tmp_path: Path):
    """Ratios below the threshold should not recommend switching references."""

    pet_mask = tmp_path / 'pet_mask.nii.gz'
    nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)).to_filename(pet_mask)

    t1_mask = tmp_path / 't1_mask.nii.gz'
    nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)).to_filename(t1_mask)

    use_nu, ratio, pet_vol, t1_vol = _detect_large_pet_mask(str(pet_mask), str(t1_mask))

    assert use_nu is False
    assert ratio == pytest.approx(1.0)
    assert pet_vol == pytest.approx(t1_vol)


def test_construct_nu_path_generates_expected_location():
    """``nu.mgz`` paths should be constructed in the standard FreeSurfer layout."""

    path = _construct_nu_path('/opt/freesurfer/subjects', 'sub-01')
    assert path.endswith('/opt/freesurfer/subjects/sub-01/mri/nu.mgz')


def test_volume_ratio_forwarded_to_summary(bids_root: Path, tmp_path: Path):
    """The PET/T1w volume ratio should flow into the report summary node."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))

    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    detect_large_mask = wf.get_node('detect_large_mask')
    summary = wf.get_node('summary')

    edge = wf._graph.get_edge_data(detect_large_mask, summary)
    assert ('volume_ratio', 'volume_ratio') in edge['connect']
    assert detect_large_mask.inputs.volume_ratio_threshold == 1.5


def test_init_refmask_report_wf(tmp_path: Path):
    """Ensure the refmask report workflow initializes without errors."""
    wf = init_refmask_report_wf(output_dir=str(tmp_path), ref_name='test')
    assert 'mask_report' in wf.list_node_names()
    ds = wf.get_node('ds_report_refmask')
    assert ds.inputs.desc == 'ref'
    assert ds.inputs.label == 'test'
    assert 'label' in ds.interface._allowed_entities
    assert ds.inputs.suffix == 'pet'


def test_reports_spec_contains_refmask():
    """Check that the report specification includes the refmask reportlet."""
    for fname in ('reports-spec.yml', 'reports-spec-pet.yml'):
        spec = yaml.safe_load((data.load.readable(fname)).read_text())
        pet_section = next(s for s in spec['sections'] if s['name'] == 'PET')
        assert any(
            r.get('bids', {}).get('desc') == 'ref' and 'label' not in r.get('bids', {})
            for r in pet_section['reportlets']
        )


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
