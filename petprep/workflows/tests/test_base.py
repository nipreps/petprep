import copy
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline.engine.utils import evaluate_connect_function, generate_expanded_graph
from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES
from niworkflows.utils.bids import collect_data as original_collect_data
from niworkflows.utils.testing import generate_bids_skeleton

from ... import config
from .. import base as base_module
from ..base import (
    _build_pvc_boilerplate,
    _build_reference_mask_boilerplate,
    _build_segmentation_boilerplate,
    _detect_existing_highres_freesurfer,
    _fix_multi_source_name,
    _format_geometry,
    _fmt_group,
    _freesurfer_subjects_dir,
    _image_geometry,
    _is_submillimeter_anat,
    _prefix,
    _session_bids_filters,
    _stringify_sessions,
    _subject_fs_id,
    _warn_about_submillimeter_recon,
    init_petprep_wf,
    init_single_subject_wf,
)
from ..tests import mock_config

BASE_LAYOUT = {
    '01': {
        'anat': [
            {'suffix': 'T1w'},
            {'suffix': 'inplaneT2'},
        ],
        'pet': [
            {
                'suffix': 'pet',
                'metadata': {},
            },
        ],
    },
}

MIXED_LAYOUT = {
    '01': {
        'anat': [{'suffix': 'T1w'}],
        'pet': [{'suffix': 'pet', 'metadata': {}}],
    },
    '02': {
        'pet': [{'suffix': 'pet', 'metadata': {}}],
    },
    '03': {
        'anat': [{'suffix': 'T1w'}],
    },
}


@pytest.fixture(scope='module')
def custom_queries():
    queries = copy.deepcopy(DEFAULT_BIDS_QUERIES)
    queries['pet'] = {'datatype': 'pet', 'suffix': 'pet'}
    queries['t1w'].pop('datatype', None)
    return queries


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
    base = tmp_path_factory.mktemp('base')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, BASE_LAYOUT)

    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    # anat files
    anat_dir = bids_dir / 'sub-01' / 'anat'
    anat_dir.mkdir(parents=True, exist_ok=True)
    img.to_filename(anat_dir / 'sub-01_T1w.nii.gz')
    img.to_filename(anat_dir / 'sub-01_inplaneT2.nii.gz')

    # pet file
    pet_dir = bids_dir / 'sub-01' / 'pet'
    pet_dir.mkdir(parents=True, exist_ok=True)
    pet_path = pet_dir / 'sub-01_pet.nii.gz'
    img.to_filename(pet_path)

    # Add metadata explicitly
    metadata = {
        'FrameTimesStart': [0],
        'FrameDuration': [1],
    }
    json_path = pet_dir / 'sub-01_pet.json'
    json_path.write_text(json.dumps(metadata))

    return bids_dir


@pytest.fixture(scope='module')
def multisession_bids_root(tmp_path_factory):
    base = tmp_path_factory.mktemp('multisession')
    bids_dir = base / 'bids'
    bids_dir.mkdir(parents=True, exist_ok=True)
    img = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))
    (bids_dir / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')
    for ses in ['01', '02']:
        anat_dir = bids_dir / 'sub-01' / f'ses-{ses}' / 'anat'
        pet_dir = bids_dir / 'sub-01' / f'ses-{ses}' / 'pet'
        anat_dir.mkdir(parents=True, exist_ok=True)
        pet_dir.mkdir(parents=True, exist_ok=True)
        img.to_filename(anat_dir / f'sub-01_ses-{ses}_T1w.nii.gz')
        pet_path = pet_dir / f'sub-01_ses-{ses}_task-rest_run-1_pet.nii.gz'
        img.to_filename(pet_path)
        (pet_path.with_suffix('').with_suffix('.json')).write_text(
            '{"FrameTimesStart": [0], "FrameDuration": [1]}'
        )
    return bids_dir


@pytest.fixture(scope='module')
def mixed_bids_root(tmp_path_factory):
    base = tmp_path_factory.mktemp('mixed-subjects')
    bids_dir = base / 'bids'
    generate_bids_skeleton(bids_dir, MIXED_LAYOUT)

    img3d = nb.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    img4d = nb.Nifti1Image(np.zeros((10, 10, 10, 10)), np.eye(4))

    anat_dir = bids_dir / 'sub-01' / 'anat'
    anat_dir.mkdir(parents=True, exist_ok=True)
    img3d.to_filename(anat_dir / 'sub-01_T1w.nii.gz')

    for subject_id in ('01', '02'):
        pet_dir = bids_dir / f'sub-{subject_id}' / 'pet'
        pet_dir.mkdir(parents=True, exist_ok=True)
        pet_path = pet_dir / f'sub-{subject_id}_pet.nii.gz'
        img4d.to_filename(pet_path)
        (pet_path.with_suffix('').with_suffix('.json')).write_text(
            json.dumps({'FrameTimesStart': [0], 'FrameDuration': [1]})
        )

    anat_dir = bids_dir / 'sub-03' / 'anat'
    anat_dir.mkdir(parents=True, exist_ok=True)
    img3d.to_filename(anat_dir / 'sub-03_T1w.nii.gz')

    return bids_dir


def test_segmentation_shared_across_runs(multisession_bids_root):
    with mock_config(bids_dir=multisession_bids_root):
        wf = init_single_subject_wf('01')
    flatgraph = wf._create_flat_graph()
    generate_expanded_graph(flatgraph)

    seg_wf_name = f'pet_{config.workflow.seg}_seg_wf'
    seg_nodes = [n for n in wf.list_node_names() if n.startswith(seg_wf_name)]
    assert seg_nodes

    pet_wf_names = [
        n
        for n in {name.split('.')[0] for name in wf.list_node_names() if name.startswith('pet_')}
        if n != seg_wf_name
    ]
    assert len(pet_wf_names) == 2

    seg_node = wf.get_node(seg_wf_name)
    for name in pet_wf_names:
        pet_node = wf.get_node(name)
        edge = wf._graph.get_edge_data(seg_node, pet_node)
        assert ('outputnode.segmentation', 'inputnode.segmentation') in edge['connect']
        assert ('outputnode.dseg_tsv', 'inputnode.dseg_tsv') in edge['connect']
        assert all('_seg_wf' not in n for n in pet_node.list_node_names())


def test_segmentation_boilerplate_mentions_atlas_reference():
    desc = _build_segmentation_boilerplate('MASSP20')
    assert 'atlas' in desc
    assert 'warped into anatomical space' in desc
    assert '[@massp20]' in desc


def test_pvc_boilerplate_includes_tool_reference():
    desc = _build_pvc_boilerplate('petpvc', 'GTM', (5.0,))
    assert '``petpvc``' in desc
    assert '``GTM``' in desc
    assert '[@petpvc]' in desc


def test_reference_mask_boilerplate_predefined():
    desc = _build_reference_mask_boilerplate('cerebellum', None)
    assert 'predefined reference region mask' in desc
    assert '``cerebellum``' in desc


def test_reference_mask_boilerplate_semiovale_citation():
    desc = _build_reference_mask_boilerplate('semiovale', None)
    assert 'centrum semiovale white matter' in desc
    assert '[@doi:10.1177/0271678X261441071]' in desc


def test_reference_mask_boilerplate_custom_labels():
    desc = _build_reference_mask_boilerplate('custom', (8, 47))
    assert 'segmentation labels (8, 47)' in desc
    assert 'time-activity curve was extracted' in desc


def test_init_petprep_wf_skips_subjects_missing_required_modalities(mixed_bids_root):
    with mock_config(bids_dir=mixed_bids_root):
        config.execution.participant_label = ['01', '02', '03']
        wf = init_petprep_wf()

    assert any(name.startswith('sub_01_wf.') for name in wf.list_node_names())
    assert not any(name.startswith('sub_02_wf.') for name in wf.list_node_names())
    assert not any(name.startswith('sub_03_wf.') for name in wf.list_node_names())


def test_init_petprep_wf_sessionwise_builds_session_workflows(multisession_bids_root, tmp_path):
    with mock_config(bids_dir=multisession_bids_root):
        config.workflow.subject_anatomical_reference = 'sessionwise'
        config.execution.bids_filters['pet'] = {'session': ['01', '02']}
        config.execution.processing_groups = [('01', '01'), ('01', '02')]
        config.execution.derivatives = {'petprep': tmp_path}
        with patch('smriprep.utils.bids.collect_derivatives', return_value={}) as collect_derivs:
            wf = init_petprep_wf()
            petprep_dir = config.execution.petprep_dir
            run_uuid = config.execution.run_uuid
            assert (petprep_dir / 'sub-01' / 'ses-01' / 'log' / run_uuid / 'petprep.toml').exists()
            assert (petprep_dir / 'sub-01' / 'ses-02' / 'log' / run_uuid / 'petprep.toml').exists()

    node_names = wf.list_node_names()
    assert any(name.startswith('sub_01_ses_01_wf.') for name in node_names)
    assert any(name.startswith('sub_01_ses_02_wf.') for name in node_names)
    assert not any(name.startswith('sub_01_wf.') for name in node_names)
    assert [call.kwargs['session_id'] for call in collect_derivs.call_args_list] == ['01', '02']
    session_node = next(
        (node for node in wf._get_all_nodes() if 'sub_01_ses_01_wf' in node.fullname),
        None,
    )
    assert session_node is not None
    assert session_node.config['execution']['crashdump_dir'] == str(
        petprep_dir / 'sub-01' / 'ses-01' / 'log' / run_uuid
    )


def test_subject_fs_id_evaluates_as_nipype_connection_function():
    source = inspect.getsource(_subject_fs_id)

    assert evaluate_connect_function(source, [None], 'sub-976') == 'sub-976'
    assert evaluate_connect_function(source, ['wave1'], '976') == 'sub-976_ses-wave1'
    assert evaluate_connect_function(source, [['ses-01', 'ses-02']], 'sub-976') == (
        'sub-976_ses-01_02'
    )


def test_fix_multi_source_name_keeps_session_only_when_requested():
    source = inspect.getsource(_fix_multi_source_name)
    t1w_files = [
        '/path/to/sub-976/ses-01/anat/sub-976_ses-01_run-1_T1w.nii.gz',
        '/path/to/sub-976/ses-01/anat/sub-976_ses-01_run-2_T1w.nii.gz',
    ]

    assert _fix_multi_source_name(t1w_files[0]) == t1w_files[0]
    assert _fix_multi_source_name(t1w_files) == ('/path/to/sub-976/ses-01/anat/sub-976_T1w.nii.gz')
    assert _fix_multi_source_name(t1w_files, 'ses-01') == (
        '/path/to/sub-976/ses-01/anat/sub-976_ses-01_T1w.nii.gz'
    )
    assert _fix_multi_source_name(t1w_files, ['ses-01', '02']) == (
        '/path/to/sub-976/ses-01/anat/sub-976_ses-01_02_T1w.nii.gz'
    )

    assert evaluate_connect_function(source, [None], t1w_files[0]) == t1w_files[0]
    assert evaluate_connect_function(source, [None], t1w_files) == (
        '/path/to/sub-976/ses-01/anat/sub-976_T1w.nii.gz'
    )
    assert evaluate_connect_function(source, ['ses-01'], t1w_files) == (
        '/path/to/sub-976/ses-01/anat/sub-976_ses-01_T1w.nii.gz'
    )
    assert evaluate_connect_function(source, [['ses-01', '02']], t1w_files) == (
        '/path/to/sub-976/ses-01/anat/sub-976_ses-01_02_T1w.nii.gz'
    )


def test_fix_multi_source_name_rejects_non_bids_name():
    with pytest.raises(AttributeError, match='Could not extract BIDS information'):
        _fix_multi_source_name(['/path/to/anat/T1w.nii.gz'])


def test_subject_id_helpers():
    assert _prefix('976') == 'sub-976'
    assert _prefix('sub-976') == 'sub-976'

    assert _subject_fs_id('976') == 'sub-976'
    assert _subject_fs_id('sub-976', None) == 'sub-976'
    assert _subject_fs_id('976', 'wave1') == 'sub-976_ses-wave1'
    assert _subject_fs_id('976', 'ses-wave1') == 'sub-976_ses-wave1'
    assert _subject_fs_id('sub-976', ['ses-01', 'ses-02']) == 'sub-976_ses-01_02'


def test_image_geometry_and_format(tmp_path):
    image = tmp_path / 'sub-01_T1w.nii.gz'
    nb.Nifti1Image(np.zeros((4, 5, 6)), np.diag([0.5, 1.0, 2.25, 1])).to_filename(image)

    shape, zooms = _image_geometry(image)

    assert shape == (4, 5, 6)
    assert zooms == (0.5, 1.0, 2.25)
    assert _format_geometry(shape, zooms) == '4 x 5 x 6 @ 0.5 x 1 x 2.25 mm'


def test_is_submillimeter_anat(tmp_path):
    submm = tmp_path / 'submm_T1w.nii.gz'
    anisotropic = tmp_path / 'anisotropic_T1w.nii.gz'

    nb.Nifti1Image(np.zeros((10, 10, 10)), np.diag([0.5, 0.5, 0.5, 1])).to_filename(submm)
    nb.Nifti1Image(np.zeros((10, 10, 10)), np.diag([0.5, 0.5, 1.0, 1])).to_filename(anisotropic)

    assert _is_submillimeter_anat(submm)
    assert not _is_submillimeter_anat(anisotropic)


def test_is_submillimeter_anat_handles_unreadable_image(bids_root, tmp_path, monkeypatch):
    messages = []

    with mock_config(bids_dir=bids_root):
        monkeypatch.setattr(config.loggers.workflow, 'warning', messages.append)

        assert not _is_submillimeter_anat(tmp_path / 'missing_T1w.nii.gz')

    assert len(messages) == 1
    assert 'Could not inspect T1w resolution' in messages[0]


def _raise_runtime_error(_image_file):
    raise RuntimeError


def test_freesurfer_subjects_dir_defaults_to_output_dir(bids_root, tmp_path):
    with mock_config(bids_dir=bids_root):
        config.execution.output_dir = tmp_path
        config.execution.fs_subjects_dir = None

        assert _freesurfer_subjects_dir() == tmp_path / 'freesurfer'

        config.execution.fs_subjects_dir = tmp_path / 'custom-fs'

        assert _freesurfer_subjects_dir() == (tmp_path / 'custom-fs').absolute()


def test_detect_existing_highres_freesurfer(bids_root, tmp_path):
    with mock_config(bids_dir=bids_root):
        config.execution.output_dir = tmp_path
        config.execution.fs_subjects_dir = tmp_path / 'freesurfer'
        mri_dir = tmp_path / 'freesurfer' / 'sub-01' / 'mri'
        mri_dir.mkdir(parents=True)
        nu = mri_dir / 'nu.mgz'
        nb.MGHImage(np.zeros((10, 10, 10), dtype='f4'), np.diag([0.5, 0.5, 0.5, 1])).to_filename(
            nu
        )

        detected = _detect_existing_highres_freesurfer('sub-01')

    assert detected is not None
    detected_file, shape, zooms = detected
    assert detected_file == nu
    assert shape == (10, 10, 10)
    assert max(zooms) == 0.5


def test_detect_existing_highres_freesurfer_ignores_standard_grid(bids_root, tmp_path):
    with mock_config(bids_dir=bids_root):
        config.execution.fs_subjects_dir = tmp_path / 'freesurfer'
        mri_dir = tmp_path / 'freesurfer' / 'sub-01' / 'mri'
        mri_dir.mkdir(parents=True)
        nb.MGHImage(np.zeros((10, 10, 10), dtype='f4'), np.eye(4)).to_filename(
            mri_dir / 'nu.mgz'
        )

        assert _detect_existing_highres_freesurfer('sub-01') is None


def test_detect_existing_highres_freesurfer_handles_unreadable_outputs(
    bids_root, tmp_path, monkeypatch
):
    with mock_config(bids_dir=bids_root):
        config.execution.fs_subjects_dir = tmp_path / 'freesurfer'
        mri_dir = tmp_path / 'freesurfer' / 'sub-01' / 'mri'
        mri_dir.mkdir(parents=True)
        (mri_dir / 'nu.mgz').write_text('not an image')
        monkeypatch.setattr(base_module, '_image_geometry', _raise_runtime_error)

        assert _detect_existing_highres_freesurfer('sub-01') is None


def test_detect_existing_highres_freesurfer_flags_large_gtmseg(
    bids_root, tmp_path, monkeypatch
):
    with mock_config(bids_dir=bids_root):
        config.execution.fs_subjects_dir = tmp_path / 'freesurfer'
        mri_dir = tmp_path / 'freesurfer' / 'sub-01' / 'mri'
        mri_dir.mkdir(parents=True)
        gtmseg = mri_dir / 'gtmseg.mgz'
        gtmseg.write_text('placeholder')

        def fake_geometry(image_file):
            assert Path(image_file).name == 'gtmseg.mgz'
            return (682, 762, 820), (0.244, 0.244, 0.244)

        monkeypatch.setattr(base_module, '_image_geometry', fake_geometry)

        detected = _detect_existing_highres_freesurfer('sub-01')

    assert detected == (gtmseg, (682, 762, 820), (0.244, 0.244, 0.244))


def test_detect_existing_highres_freesurfer_handles_unreadable_gtmseg(
    bids_root, tmp_path, monkeypatch
):
    with mock_config(bids_dir=bids_root):
        config.execution.fs_subjects_dir = tmp_path / 'freesurfer'
        mri_dir = tmp_path / 'freesurfer' / 'sub-01' / 'mri'
        mri_dir.mkdir(parents=True)
        (mri_dir / 'gtmseg.mgz').write_text('not an image')
        monkeypatch.setattr(base_module, '_image_geometry', _raise_runtime_error)

        assert _detect_existing_highres_freesurfer('sub-01') is None


def test_warn_about_submillimeter_recon_branches(bids_root, tmp_path, monkeypatch):
    messages = []
    submm = tmp_path / 'submm_T1w.nii.gz'
    standard = tmp_path / 'standard_T1w.nii.gz'
    nb.Nifti1Image(np.zeros((10, 10, 10)), np.diag([0.5, 0.5, 0.5, 1])).to_filename(submm)
    nb.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4)).to_filename(standard)

    with mock_config(bids_dir=bids_root):
        config.workflow.run_reconall = False
        monkeypatch.setattr(config.loggers.workflow, 'warning', messages.append)

        _warn_about_submillimeter_recon(
            subject_id='01', session_id=None, t1w_files=[submm], pet_runs=['pet1']
        )

        assert messages == []

        config.workflow.run_reconall = True
        config.execution.fs_subjects_dir = tmp_path / 'empty-fs'
        config.workflow.hires = False

        _warn_about_submillimeter_recon(
            subject_id='01', session_id=None, t1w_files=[], pet_runs=['pet1']
        )
        _warn_about_submillimeter_recon(
            subject_id='01', session_id=None, t1w_files=[standard], pet_runs=['pet1']
        )

        assert messages == []

        _warn_about_submillimeter_recon(
            subject_id='01', session_id=None, t1w_files=[submm], pet_runs=['pet1']
        )

        assert len(messages) == 1
        assert 'will run FreeSurfer without submillimeter reconstruction' in messages[-1]

        config.workflow.hires = True
        _warn_about_submillimeter_recon(
            subject_id='01', session_id='ses-01', t1w_files=[submm], pet_runs=['pet1', 'pet2']
        )

        assert len(messages) == 2
        assert 'Submillimeter FreeSurfer reconstruction is enabled' in messages[-1]
        assert '2 PET run(s)' in messages[-1]


def test_warn_about_submillimeter_recon_existing_highres(bids_root, tmp_path, monkeypatch):
    messages = []

    with mock_config(bids_dir=bids_root):
        config.execution.fs_subjects_dir = tmp_path / 'freesurfer'
        config.workflow.run_reconall = True
        monkeypatch.setattr(config.loggers.workflow, 'warning', messages.append)
        mri_dir = tmp_path / 'freesurfer' / 'sub-01' / 'mri'
        mri_dir.mkdir(parents=True)
        nb.MGHImage(np.zeros((10, 10, 10), dtype='f4'), np.diag([0.5, 0.5, 0.5, 1])).to_filename(
            mri_dir / 'nu.mgz'
        )

        _warn_about_submillimeter_recon(
            subject_id='01', session_id=None, t1w_files=[], pet_runs=['pet1']
        )

    assert len(messages) == 1
    assert 'Existing high-resolution FreeSurfer outputs were detected' in messages[0]
    assert '--no-submm-recon only affects new recon-all runs' in messages[0]


def test_warn_about_submillimeter_recon_handles_geometry_failure(
    bids_root, tmp_path, monkeypatch
):
    messages = []
    submm = tmp_path / 'submm_T1w.nii.gz'
    submm.write_text('placeholder')

    with mock_config(bids_dir=bids_root):
        config.workflow.run_reconall = True
        config.workflow.hires = False
        config.execution.fs_subjects_dir = tmp_path / 'empty-fs'
        monkeypatch.setattr(config.loggers.workflow, 'warning', messages.append)
        monkeypatch.setattr(base_module, '_is_submillimeter_anat', lambda _image_file: True)
        monkeypatch.setattr(base_module, '_image_geometry', _raise_runtime_error)

        _warn_about_submillimeter_recon(
            subject_id='01', session_id=None, t1w_files=[submm], pet_runs=[]
        )

    assert len(messages) == 1
    assert f'Submillimeter T1w image detected for sub-01: {submm}' in messages[0]


def test_session_helpers_format_groups_and_bids_filters(bids_root):
    with mock_config(bids_dir=bids_root):
        config.execution.bids_filters = {
            'pet': {'task': 'rest'},
            't1w': {'suffix': 'T1w'},
            'dwi': {'session': 'keep'},
        }

        assert _stringify_sessions(None) is None
        assert _fmt_group('01') == 'sub-01'
        assert _fmt_group('01', ['ses-pre', 'ses-post']) == 'sub-01/ses-pre_post'
        assert _session_bids_filters('ses-pre') == config.execution.bids_filters

        config.workflow.subject_anatomical_reference = 'sessionwise'
        filters = _session_bids_filters('ses-pre')

    assert config.execution.bids_filters['pet'] == {'task': 'rest'}
    assert filters['pet'] == {'task': 'rest', 'session': 'ses-pre'}
    assert filters['t1w'] == {'suffix': 'T1w', 'session': 'ses-pre'}
    assert filters['t2w'] == {'session': 'ses-pre'}
    assert filters['flair'] == {'session': 'ses-pre'}
    assert filters['roi'] == {'session': 'ses-pre'}
    assert filters['dwi'] == {'session': 'keep'}


def _make_params(
    pet2anat_init: str = 'auto',
    medial_surface_nan: bool = False,
    cifti_output: bool | str = False,
    run_msmsulc: bool = True,
    skull_strip_t1w: str = 'auto',
    freesurfer: bool = True,
    ignore: list[str] = None,
    force: list[str] = None,
    bids_filters: dict = None,
):
    if ignore is None:
        ignore = []
    if force is None:
        force = []
    if bids_filters is None:
        bids_filters = {}
    return (
        pet2anat_init,
        medial_surface_nan,
        cifti_output,
        run_msmsulc,
        skull_strip_t1w,
        freesurfer,
        ignore,
        force,
        bids_filters,
    )


@pytest.mark.parametrize('level', ['minimal', 'resampling', 'full'])
@pytest.mark.parametrize('anat_only', [False, True])
@pytest.mark.parametrize(
    (
        'pet2anat_init',
        'medial_surface_nan',
        'cifti_output',
        'run_msmsulc',
        'skull_strip_t1w',
        'freesurfer',
        'ignore',
        'force',
        'bids_filters',
    ),
    [
        _make_params(),
        _make_params(pet2anat_init='t1w'),
        _make_params(pet2anat_init='t2w'),
        _make_params(pet2anat_init='header'),
        _make_params(force=['bbr']),
        _make_params(force=['no-bbr']),
        _make_params(pet2anat_init='header', force=['bbr']),
        _make_params(medial_surface_nan=True),
        _make_params(cifti_output='91k'),
        _make_params(cifti_output='91k', run_msmsulc=False),
        _make_params(skull_strip_t1w='force'),
        _make_params(skull_strip_t1w='skip'),
        _make_params(freesurfer=False),
        _make_params(freesurfer=False, force=['bbr']),
        _make_params(freesurfer=False, force=['no-bbr']),
    ],
)
def test_init_petprep_wf(
    bids_root: Path,
    tmp_path: Path,
    level: str,
    anat_only: bool,
    pet2anat_init: str,
    medial_surface_nan: bool,
    cifti_output: bool | str,
    run_msmsulc: bool,
    skull_strip_t1w: str,
    freesurfer: bool,
    ignore: list[str],
    force: list[str],
    bids_filters: dict,
    custom_queries: dict,
):
    with mock_config(bids_dir=bids_root):
        config.workflow.level = level
        config.workflow.anat_only = anat_only
        config.workflow.pet2anat_init = pet2anat_init
        config.workflow.medial_surface_nan = medial_surface_nan
        config.workflow.run_msmsulc = run_msmsulc
        config.workflow.skull_strip_t1w = skull_strip_t1w
        config.workflow.cifti_output = cifti_output
        config.workflow.run_reconall = freesurfer
        config.workflow.ignore = ignore
        config.workflow.force = force

        with patch.dict('petprep.config.execution.bids_filters', bids_filters):
            # Patch the correct function with the correct return value explicitly
            with patch('niworkflows.utils.bids.collect_data') as mock_collect_data:
                params = inspect.signature(original_collect_data).parameters
                kwargs = {
                    'bids_filters': bids_filters,
                    'queries': custom_queries,
                }
                if 'require_pet' in params:
                    kwargs['require_pet'] = True

                mock_collect_data.return_value = original_collect_data(
                    bids_root,
                    '01',
                    **kwargs,
                )

                wf = init_petprep_wf()

    generate_expanded_graph(wf._create_flat_graph())
