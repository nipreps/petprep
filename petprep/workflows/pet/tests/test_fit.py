import json
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
from ....interfaces.registration import PETCoregistrationFallback
from ....utils import bids
from ...tests import mock_config
from ...tests.test_base import BASE_LAYOUT
from .. import fit as pet_fit
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
from ..registration import init_pet_reg_wf


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
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))

    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}'
        )

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
    assert summary.inputs.requested_anatref == 'auto'
    assert summary.inputs.hmc_disabled is True


def test_petref_auto_uses_template_for_3d_pet(bids_root: Path, tmp_path: Path):
    """3D PET data should not fan out into redundant auto reference candidates."""
    from ....utils.misc import estimate_pet_mem_usage

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0], "FrameDuration": [1]}')

    estimate_pet_mem_usage.cache_clear()
    try:
        with mock_config(bids_dir=bids_root):
            config.workflow.petref = 'auto'
            wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)
    finally:
        estimate_pet_mem_usage.cache_clear()

    node_names = wf.list_node_names()
    assert 'petref_candidates' not in node_names
    assert 'select_best_petref' not in node_names
    assert 'auto_twa_reference' not in node_names
    assert 'auto_sum_reference' not in node_names
    assert 'auto_first5min_reference' not in node_names
    assert any(name.startswith('pet_reg_wf.') for name in node_names)
    assert not any(name.startswith('pet_reg_wf_') for name in node_names)

    petref_buffer = wf.get_node('petref_buffer')
    assert petref_buffer.inputs.petref == pet_series[0]

    summary = wf.get_node('summary')
    assert summary.inputs.petref_strategy == 'template'
    assert summary.inputs.requested_petref_strategy == 'auto'


def test_petref_auto_mixed_3d_and_4d_pet_runs(bids_root: Path, tmp_path: Path):
    """The 3D auto shortcut is decided per PET workflow, not globally."""
    from ....utils.misc import estimate_pet_mem_usage

    pet_dir = bids_root / 'sub-01' / 'pet'
    pet_3d = pet_dir / 'sub-01_task-rest_run-1_pet.nii.gz'
    pet_4d = pet_dir / 'sub-01_task-rest_run-2_pet.nii.gz'

    nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(pet_3d)
    nb.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4)).to_filename(pet_4d)
    pet_3d.with_suffix('').with_suffix('.json').write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )
    pet_4d.with_suffix('').with_suffix('.json').write_text(
        '{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}'
    )

    estimate_pet_mem_usage.cache_clear()
    try:
        with mock_config(bids_dir=bids_root):
            config.workflow.petref = 'auto'
            wf_3d = init_pet_fit_wf(pet_series=[str(pet_3d)], precomputed={}, omp_nthreads=1)
            wf_4d = init_pet_fit_wf(pet_series=[str(pet_4d)], precomputed={}, omp_nthreads=1)
    finally:
        estimate_pet_mem_usage.cache_clear()

    node_names_3d = wf_3d.list_node_names()
    node_names_4d = wf_4d.list_node_names()

    assert 'petref_candidates' not in node_names_3d
    assert 'select_best_petref' not in node_names_3d
    assert 'petref_candidates' in node_names_4d
    assert 'select_best_petref' in node_names_4d

    assert wf_3d.get_node('summary').inputs.petref_strategy == 'template'
    assert wf_4d.get_node('summary').inputs.petref_strategy == 'auto'


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
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
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
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.hmc_off = False
        config.workflow.petref = 'template'
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    assert any(name.startswith('pet_hmc_wf') for name in wf.list_node_names())

    dummy_affine = tmp_path / 'xfm.txt'
    np.savetxt(dummy_affine, np.eye(4))
    ref_file = tmp_path / 'ref.nii'
    img.to_filename(ref_file)
    precomputed = {'petref': str(ref_file), 'transforms': {'hmc': str(dummy_affine)}}

    with mock_config(bids_dir=bids_root):
        config.workflow.hmc_off = False
        config.workflow.petref = 'template'
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
    assert any(name.endswith('.mri_robust_register') for name in node_names)
    assert not any(name.endswith('.mri_coreg') for name in node_names)
    assert not any(name.endswith('.ants_registration') for name in node_names)


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
    assert any(name.endswith('.ants_registration') for name in node_names)
    assert not any(name.endswith('.mri_coreg') for name in node_names)
    assert not any(name.endswith('.mri_robust_register') for name in node_names)


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
    assert any(name.endswith('.ants_registration') for name in node_names)
    assert any(name.endswith('.mri_coreg') for name in node_names)
    assert any(name.endswith('.select_best') for name in node_names)
    assert any(name.endswith('.score_ants') for name in node_names)
    assert any(name.endswith('.score_fs') for name in node_names)
    assert any(name.endswith('.warp_pet_ants') for name in node_names)
    assert any(name.endswith('.warp_pet_fs') for name in node_names)


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
        config.workflow.petref = 'auto'
        config.workflow.pet2anat_method = 'mri_coreg'
        config.workflow.petref_specified = True
        config.workflow.pet2anat_method_specified = True
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed=precomputed, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert any(name.endswith('.mri_coreg') for name in node_names)
    assert wf.get_node('outputnode').inputs.petref2anat_xfm is Undefined


def test_pet_reg_no_crop_removes_robust_fov():
    """Disabling anatomical cropping should bypass the robustfov node."""

    wf = init_pet_reg_wf(
        pet2anat_dof=6,
        mem_gb=1,
        omp_nthreads=1,
        pet2anat_method='mri_coreg',
        crop_anat=False,
    )

    node_names = wf.list_node_names()
    assert 'robust_fov' not in node_names
    assert 'convert_anat' in node_names
    assert 'crop_anat_mask' in node_names


def _touch(path):
    path.write_text('x')
    return str(path)


def _fallback_interface(tmp_path, **inputs):
    base_inputs = {
        'ref_pet_brain': _touch(tmp_path / 'petref.nii.gz'),
        'anat_preproc': _touch(tmp_path / 'anat.nii.gz'),
        'anat_mask': _touch(tmp_path / 'mask.nii.gz'),
        'fallback_threshold': -0.05,
        'pet2anat_dof': 6,
        'pet2anat_method': 'mri_coreg',
        'mem_gb': 1.0,
        'omp_nthreads': 1,
    }
    base_inputs.update(inputs)
    return PETCoregistrationFallback(**base_inputs)


def test_pet_coreg_fallback_keeps_good_cropped_score(monkeypatch, tmp_path):
    """Acceptable cropped registration should return before running fallback."""

    monkeypatch.chdir(tmp_path)

    def _unexpected_fallback(*args, **kwargs):
        raise AssertionError('Fallback should not run when cropped score passes.')

    monkeypatch.setattr(PETCoregistrationFallback, '_run_uncropped_fallback', _unexpected_fallback)

    cropped = _touch(tmp_path / 'cropped.txt')
    cropped_inv = _touch(tmp_path / 'cropped_inv.txt')
    result = _fallback_interface(
        tmp_path,
        cropped_transform=cropped,
        cropped_inv_transform=cropped_inv,
        cropped_winner='freesurfer',
        cropped_score=-0.15,
    ).run()

    assert result.outputs.best_transform == cropped
    assert result.outputs.best_inv_transform == cropped_inv
    assert result.outputs.best_winner == 'freesurfer'
    assert result.outputs.best_score == -0.15
    assert result.outputs.fallback is False
    assert result.outputs.anat_reference == 'cropped'


def test_pet_coreg_fallback_runs_when_cropped_score_is_weak(monkeypatch, tmp_path):
    """Weak cropped registration should run uncropped fallback and keep better score."""

    monkeypatch.chdir(tmp_path)
    calls = []

    def _fake_fallback(self, cwd):
        calls.append((self.inputs.ref_pet_brain, self.inputs.anat_preproc, self.inputs.anat_mask))
        return (
            _touch(tmp_path / 'uncropped.txt'),
            _touch(tmp_path / 'uncropped_inv.txt'),
            'ants',
            -0.15,
        )

    monkeypatch.setattr(PETCoregistrationFallback, '_run_uncropped_fallback', _fake_fallback)

    result = _fallback_interface(
        tmp_path,
        pet2anat_method='auto',
        cropped_ants_transform=_touch(tmp_path / 'cropped_ants.txt'),
        cropped_fs_transform=_touch(tmp_path / 'cropped_fs.txt'),
        cropped_ants_inv_transform=_touch(tmp_path / 'cropped_ants_inv.txt'),
        cropped_fs_inv_transform=_touch(tmp_path / 'cropped_fs_inv.txt'),
        cropped_ants_score=-0.01,
        cropped_fs_score=-0.02,
        sloppy=True,
    ).run()

    assert len(calls) == 1
    assert result.outputs.best_winner == 'ants'
    assert result.outputs.best_score == -0.15
    assert result.outputs.fallback is True
    assert result.outputs.anat_reference == 'uncropped'
    assert result.outputs.registration_winner == 'ants'
    assert result.outputs.registration_score == -0.15


class _FakeFallbackWorkflow:
    def __init__(self, calls):
        self.inputs = type('inputs', (), {})()
        self.inputs.inputnode = type('inputnode', (), {})()
        self.calls = calls
        self.base_dir = None
        self.name = 'pet_reg_uncropped_fallback_wf'

    def run(self, plugin):
        self.calls['plugin'] = plugin
        self.calls['base_dir'] = self.base_dir
        self.calls['inputs'] = (
            self.inputs.inputnode.ref_pet_brain,
            self.inputs.inputnode.anat_preproc,
            self.inputs.inputnode.anat_mask,
        )
        return None


def test_pet_coreg_fallback_interface_runs_uncropped_workflow(monkeypatch, tmp_path):
    """Fallback interface should run an uncropped registration workflow lazily."""

    calls = {}

    def _fake_init_pet_reg_wf(**kwargs):
        calls['kwargs'] = kwargs
        return _FakeFallbackWorkflow(calls)

    monkeypatch.setattr(
        'petprep.workflows.pet.registration.init_pet_reg_wf',
        _fake_init_pet_reg_wf,
    )
    ants_xfm = _touch(tmp_path / 'uncropped_ants.txt')
    ants_inv = _touch(tmp_path / 'uncropped_ants_inv.txt')
    fs_xfm = _touch(tmp_path / 'uncropped_fs.txt')
    fs_inv = _touch(tmp_path / 'uncropped_fs_inv.txt')

    def _fake_result(**outputs):
        return type(
            'result',
            (),
            {
                'outputs': type('outputs', (), outputs)(),
            },
        )()

    def _fake_loadpkl(path):
        if 'convert_xfm_ants' in path:
            return _fake_result(out_xfm=ants_xfm, out_inv=ants_inv)
        if 'convert_xfm_fs' in path:
            return _fake_result(out_xfm=fs_xfm, out_inv=fs_inv)
        if 'score_ants' in path:
            return _fake_result(similarity=-0.15)
        if 'score_fs' in path:
            return _fake_result(similarity=-0.01)
        raise AssertionError(f'Unexpected pickle path: {path}')  # pragma: no cover

    monkeypatch.setattr('nipype.utils.filemanip.loadpkl', _fake_loadpkl)

    interface = _fallback_interface(
        tmp_path,
        pet2anat_method='auto',
        cropped_ants_transform=_touch(tmp_path / 'cropped_ants.txt'),
        cropped_fs_transform=_touch(tmp_path / 'cropped_fs.txt'),
        cropped_ants_inv_transform=_touch(tmp_path / 'cropped_ants_inv.txt'),
        cropped_fs_inv_transform=_touch(tmp_path / 'cropped_fs_inv.txt'),
        cropped_ants_score=-0.01,
        cropped_fs_score=-0.02,
        mem_gb=1.5,
        omp_nthreads=2,
        sloppy=True,
    )

    assert interface._run_uncropped_fallback(str(tmp_path)) == (
        ants_xfm,
        ants_inv,
        'ants',
        -0.15,
    )
    assert calls['plugin'] == 'Linear'
    assert calls['inputs'] == (
        interface.inputs.ref_pet_brain,
        interface.inputs.anat_preproc,
        interface.inputs.anat_mask,
    )
    assert calls['kwargs'] == {
        'pet2anat_dof': 6,
        'mem_gb': 1.5,
        'omp_nthreads': 2,
        'pet2anat_method': 'auto',
        'crop_anat': False,
        'sloppy': True,
        'name': 'pet_reg_uncropped_fallback_wf',
    }


def test_pet_coreg_fallback_interface_reads_manual_uncropped_outputs(monkeypatch, tmp_path):
    """Manual fallback should load transform and score outputs from the uncropped workflow."""

    monkeypatch.setattr(
        'petprep.workflows.pet.registration.init_pet_reg_wf',
        lambda **kwargs: _FakeFallbackWorkflow({}),
    )
    xfm = _touch(tmp_path / 'uncropped.txt')
    inv_xfm = _touch(tmp_path / 'uncropped_inv.txt')

    def _fake_result(**outputs):
        return type(
            'result',
            (),
            {
                'outputs': type('outputs', (), outputs)(),
            },
        )()

    def _fake_loadpkl(path):
        if 'convert_xfm' in path:
            return _fake_result(out_xfm=xfm, out_inv=inv_xfm)
        if 'score_registration' in path:
            return _fake_result(similarity=-0.12)
        raise AssertionError(f'Unexpected pickle path: {path}')  # pragma: no cover

    monkeypatch.setattr('nipype.utils.filemanip.loadpkl', _fake_loadpkl)

    interface = _fallback_interface(
        tmp_path,
        cropped_transform=_touch(tmp_path / 'cropped.txt'),
        cropped_inv_transform=_touch(tmp_path / 'cropped_inv.txt'),
        cropped_score=-0.01,
    )

    assert interface._run_uncropped_fallback(str(tmp_path)) == (xfm, inv_xfm, None, -0.12)
    assert interface._score_summary['uncropped'] == {
        'mri_coreg': -0.12,
        'winner': None,
        'score': -0.12,
    }


def test_pet_coreg_fallback_populates_freesurfer_outputs_without_inverse(monkeypatch, tmp_path):
    """FreeSurfer fallback should report score and synthesize a missing inverse."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        'petprep.workflows.pet.registration.init_pet_reg_wf',
        lambda **kwargs: _FakeFallbackWorkflow({}),
    )

    ants_xfm = _touch(tmp_path / 'uncropped_ants.txt')
    ants_inv = _touch(tmp_path / 'uncropped_ants_inv.txt')
    fs_xfm = _touch(tmp_path / 'uncropped_fs.txt')
    synthetic_inv = _touch(tmp_path / 'synth_inv.txt')

    def _fake_result(**outputs):
        return type(
            'result',
            (),
            {
                'outputs': type('outputs', (), outputs)(),
            },
        )()

    def _fake_loadpkl(path):
        if 'convert_xfm_ants' in path:
            return _fake_result(out_xfm=ants_xfm, out_inv=ants_inv)
        if 'convert_xfm_fs' in path:
            return _fake_result(out_xfm=fs_xfm, out_inv=Undefined)
        if 'score_ants' in path:
            return _fake_result(similarity=-0.01)
        if 'score_fs' in path:
            return _fake_result(similarity=-0.2)
        raise AssertionError(f'Unexpected pickle path: {path}')  # pragma: no cover

    monkeypatch.setattr('nipype.utils.filemanip.loadpkl', _fake_loadpkl)
    monkeypatch.setattr(
        PETCoregistrationFallback,
        '_ensure_inverse_transform',
        lambda self, xfm, inv_xfm: synthetic_inv,
    )

    result = _fallback_interface(
        tmp_path,
        pet2anat_method='auto',
        cropped_ants_transform=_touch(tmp_path / 'cropped_ants.txt'),
        cropped_fs_transform=_touch(tmp_path / 'cropped_fs.txt'),
        cropped_ants_inv_transform=_touch(tmp_path / 'cropped_ants_inv.txt'),
        cropped_fs_inv_transform=_touch(tmp_path / 'cropped_fs_inv.txt'),
        cropped_ants_score=-0.01,
        cropped_fs_score=-0.02,
    ).run()

    assert Path(result.outputs.best_transform).name == 'best_transform.txt'
    assert Path(result.outputs.best_inv_transform).name == 'best_inv_transform.txt'
    assert Path(result.outputs.best_transform).read_text() == Path(fs_xfm).read_text()
    assert Path(result.outputs.best_inv_transform).read_text() == Path(synthetic_inv).read_text()
    assert result.outputs.best_winner == 'freesurfer'
    assert result.outputs.best_score == -0.2
    assert result.outputs.registration_winner == 'freesurfer'
    assert result.outputs.registration_score == -0.2
    assert result.outputs.fallback is True
    assert result.outputs.anat_reference == 'uncropped'

    scores = json.loads(Path(result.outputs.fallback_scores).read_text())
    assert scores['cropped'] == {
        'ants': -0.01,
        'freesurfer': -0.02,
        'score': -0.02,
        'winner': 'freesurfer',
    }
    assert scores['uncropped'] == {
        'ants': -0.01,
        'freesurfer': -0.2,
        'score': -0.2,
        'winner': 'freesurfer',
    }
    assert scores['selected'] == {
        'anat_reference': 'uncropped',
        'fallback': True,
        'score': -0.2,
        'winner': 'freesurfer',
    }


def test_pet_coreg_fallback_reuses_existing_inverse(tmp_path):
    """Existing inverse transform outputs should be returned without synthesis."""

    xfm = _touch(tmp_path / 'xfm.txt')
    inv_xfm = _touch(tmp_path / 'inv_xfm.txt')

    interface = _fallback_interface(tmp_path)

    assert interface._ensure_inverse_transform(xfm, inv_xfm) == inv_xfm


def test_pet_coreg_fallback_synthesizes_missing_inverse(tmp_path):
    """Missing inverse transform outputs should be generated from the forward transform."""

    xfm = tmp_path / 'xfm.tfm'
    nt.linear.Affine(
        np.array(
            [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    ).to_filename(xfm, fmt='itk')

    interface = _fallback_interface(tmp_path)
    interface._runtime_cwd = str(tmp_path)

    inv_xfm = interface._ensure_inverse_transform(str(xfm), Undefined)

    assert Path(inv_xfm) == tmp_path / 'out_inv.tfm'
    assert Path(inv_xfm).exists()
    assert np.allclose(
        nt.linear.load(inv_xfm, fmt='itk').matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 1.0, 0.0, -3.0],
                [0.0, 0.0, 1.0, -4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    ('xfm', 'inv_xfm', 'score', 'message'),
    [
        (Undefined, 'inv_xfm.txt', -0.1, 'best_transform'),
        ('xfm.txt', Undefined, -0.1, 'best_inv_transform'),
        ('xfm.txt', 'inv_xfm.txt', None, 'best_score'),
        (Undefined, Undefined, None, 'best_transform, best_inv_transform, best_score'),
    ],
)
def test_pet_coreg_fallback_rejects_incomplete_selected_outputs(
    tmp_path, xfm, inv_xfm, score, message
):
    """Fallback selection should fail loudly when the chosen result is incomplete."""

    interface = _fallback_interface(tmp_path)

    with pytest.raises(ValueError, match=message):
        interface._require_selected_outputs(xfm, inv_xfm, score)


def test_pet_coreg_fallback_keeps_cropped_when_uncropped_is_worse(monkeypatch, tmp_path):
    """Weak cropped registration should still win if uncropped score is worse."""

    monkeypatch.chdir(tmp_path)

    def _fake_fallback(self, cwd):
        return (
            _touch(tmp_path / 'uncropped.txt'),
            _touch(tmp_path / 'uncropped_inv.txt'),
            'ants',
            -0.005,
        )

    monkeypatch.setattr(PETCoregistrationFallback, '_run_uncropped_fallback', _fake_fallback)

    cropped = _touch(tmp_path / 'cropped.txt')
    cropped_inv = _touch(tmp_path / 'cropped_inv.txt')
    result = _fallback_interface(
        tmp_path,
        cropped_transform=cropped,
        cropped_inv_transform=cropped_inv,
        cropped_winner='freesurfer',
        cropped_score=-0.01,
    ).run()

    assert result.outputs.best_transform == cropped
    assert result.outputs.best_inv_transform == cropped_inv
    assert result.outputs.fallback is False
    assert result.outputs.anat_reference == 'cropped'


@pytest.mark.parametrize(
    ('ants_score', 'fs_score', 'expected_winner'),
    [
        (-0.15, -0.01, 'ants'),
        (-0.01, -0.15, 'freesurfer'),
    ],
)
def test_pet_coreg_auto_fallback_keeps_cropped_when_one_backend_passes(
    monkeypatch, tmp_path, ants_score, fs_score, expected_winner
):
    """Auto fallback should stop when either cropped backend score is acceptable."""

    monkeypatch.chdir(tmp_path)

    def _unexpected_fallback(*args, **kwargs):
        raise AssertionError('Fallback should not run when one cropped backend passes.')

    monkeypatch.setattr(PETCoregistrationFallback, '_run_uncropped_fallback', _unexpected_fallback)

    result = _fallback_interface(
        tmp_path,
        pet2anat_method='auto',
        cropped_ants_transform=_touch(tmp_path / 'cropped_ants.txt'),
        cropped_fs_transform=_touch(tmp_path / 'cropped_fs.txt'),
        cropped_ants_inv_transform=_touch(tmp_path / 'cropped_ants_inv.txt'),
        cropped_fs_inv_transform=_touch(tmp_path / 'cropped_fs_inv.txt'),
        cropped_ants_score=ants_score,
        cropped_fs_score=fs_score,
    ).run()

    assert result.outputs.best_winner == expected_winner
    assert result.outputs.best_score == min(ants_score, fs_score)
    assert result.outputs.fallback is False
    assert result.outputs.anat_reference == 'cropped'


def test_pet_fit_no_crop_reruns_coreg(bids_root: Path, tmp_path: Path):
    """Explicitly disabling crop should re-run registration and propagate to the graph."""

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
        config.workflow.pet2anat_crop = False
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed=precomputed, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert any(
        name.startswith('pet_reg_wf') and name.endswith('.mri_coreg') for name in node_names
    )
    assert not any(
        name.startswith('pet_reg_wf') and name.endswith('.robust_fov') for name in node_names
    )
    assert wf.get_node('outputnode').inputs.petref2anat_xfm is Undefined


def test_pet_fit_adds_uncropped_fallback_selector_by_default(bids_root: Path, tmp_path: Path):
    """Default cropped registration should add the uncropped fallback selector."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert any(
        name.startswith('pet_reg_wf') and name.endswith('.robust_fov') for name in node_names
    )
    assert any(name.startswith('select_crop_fallback') for name in node_names)
    assert not any(name.startswith('pet_reg_wf_no_crop') for name in node_names)

    provenance_nodes = [
        name
        for name in node_names
        if name.startswith('select_crop_fallback') and name.endswith('_provenance')
    ]
    assert provenance_nodes
    for provenance_name in provenance_nodes:
        selector_name = provenance_name.removesuffix('_provenance')
        assert selector_name in node_names
        edge = wf._graph.get_edge_data(
            wf.get_node(selector_name),
            wf.get_node(provenance_name),
        )
        assert ('best_inv_transform', 'best_inv_transform') in edge['connect']
        assert ('best_score', 'best_score') in edge['connect']
        assert ('registration_winner', 'registration_winner') in edge['connect']
        assert ('registration_score', 'registration_score') in edge['connect']
        assert ('fallback_scores', 'fallback_scores') in edge['connect']


def test_pet_fit_omits_uncropped_fallback_selector_when_disabled(bids_root: Path, tmp_path: Path):
    """Disabling the fallback should keep cropped registration as a simple graph."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.pet2anat_crop_fallback = False
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

    node_names = wf.list_node_names()
    assert any(
        name.startswith('pet_reg_wf') and name.endswith('.robust_fov') for name in node_names
    )
    assert not any(name.startswith('select_crop_fallback') for name in node_names)


def test_pet_fit_omits_uncropped_fallback_selector_for_manual_method(
    bids_root: Path, tmp_path: Path
):
    """Manual PET-to-anatomical registration should keep the requested cropped method."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 1)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.petref = 'template'
        config.workflow.pet2anat_method = 'mri_coreg'
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=2)

    node_names = wf.list_node_names()
    assert 'select_crop_fallback' not in node_names
    assert 'pet_reg_wf.robust_fov' in node_names
    assert any(
        name.startswith('pet_reg_wf') and name.endswith('.mri_coreg') for name in node_names
    )


def test_pet_fit_auto_petrefs_omit_uncropped_fallback_selector_for_manual_method(
    bids_root: Path, tmp_path: Path
):
    """Manual PET-to-anatomical registration should not add fallback for auto PET refs."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2)), np.eye(4))
    for path in pet_series:
        img.to_filename(path)
        Path(path).with_suffix('').with_suffix('.json').write_text(
            '{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}'
        )

    with mock_config(bids_dir=bids_root):
        config.workflow.petref = 'auto'
        config.workflow.pet2anat_method = 'mri_coreg'
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=2)

    node_names = wf.list_node_names()
    assert not any(name.startswith('select_crop_fallback') for name in node_names)
    for label in ('template', 'twa', 'sum', 'first5min'):
        assert f'pet_reg_wf_{label}.mri_coreg' in node_names


def test_pet_fit_hmc_off_disables_stage1(bids_root: Path, tmp_path: Path, monkeypatch):
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

    identity_xform_frames = []
    write_identity_xforms = pet_fit._write_identity_xforms

    def _record_identity_xforms(num_frames, filename):
        identity_xform_frames.append(num_frames)
        return write_identity_xforms(num_frames, filename)

    monkeypatch.setattr(pet_fit, '_write_identity_xforms', _record_identity_xforms)

    with mock_config(bids_dir=bids_root):
        config.workflow.hmc_off = True
        wf = init_pet_fit_wf(pet_series=pet_series, precomputed={}, omp_nthreads=1)

        assert not any(name.startswith('pet_hmc_wf') for name in wf.list_node_names())
        hmc_buffer = wf.get_node('hmc_buffer')
        assert str(hmc_buffer.inputs.hmc_xforms).endswith('idmat.tfm')
        assert Path(hmc_buffer.inputs.hmc_xforms).exists()
        assert identity_xform_frames == [data.shape[-1]]
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


def test_pet_fit_picks_single_precomputed_derivative(bids_root: Path, tmp_path: Path):
    """When multiple cached derivatives are present, pick the first one."""

    pet_series = [str(bids_root / 'sub-01' / 'pet' / 'sub-01_task-rest_run-1_pet.nii.gz')]
    img = nb.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4))
    for path in pet_series:
        img.to_filename(path)

    sidecar = Path(pet_series[0]).with_suffix('').with_suffix('.json')
    sidecar.write_text('{"FrameTimesStart": [0, 1], "FrameDuration": [1, 1]}')

    petrefs = [tmp_path / 'petref_a.nii.gz', tmp_path / 'petref_b.nii.gz']
    hmc_list = [tmp_path / 'hmc_a.txt', tmp_path / 'hmc_b.txt']
    petref2anat_list = [tmp_path / 'petref2anat_a.txt', tmp_path / 'petref2anat_b.txt']

    for path in petrefs:
        img.to_filename(path)
    for path in hmc_list + petref2anat_list:
        np.savetxt(path, np.eye(4))

    with mock_config(bids_dir=bids_root):
        wf = init_pet_fit_wf(
            pet_series=pet_series,
            precomputed={
                'petref': [str(p) for p in petrefs],
                'transforms': {
                    'hmc': [str(p) for p in hmc_list],
                    'petref2anat': [str(p) for p in petref2anat_list],
                },
            },
            omp_nthreads=1,
        )

    petref_buffer = wf.get_node('petref_buffer')
    hmc_buffer = wf.get_node('hmc_buffer')
    outputnode = wf.get_node('outputnode')

    assert petref_buffer.inputs.petref == str(petrefs[0])
    assert hmc_buffer.inputs.hmc_xforms == str(hmc_list[0])
    assert outputnode.inputs.petref2anat_xfm == str(petref2anat_list[0])


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
