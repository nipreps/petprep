"""PET-only modality selection, shared components, and registration contracts."""

import json
import shutil
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from petprep import config
from petprep.cli.parser import parse_args
from petprep.interfaces.tacs import ExtractTACs
from petprep.tests.test_config import _reset_config
from petprep.utils import atlas
from petprep.utils.pet_only import prepare_reference, resolve_resources, template_segmentation
from petprep.utils.pet_registration import registration_initializations, registration_masks
from petprep.workflows.base import init_petprep_wf
from petprep.workflows.pet.normalization import init_pet_template_reg_wf


def _write_motion_cache(directory, source, nframes=3, desc=True):
    """Write the motion derivatives produced by the existing anatomical workflow."""
    import nitransforms as nt

    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(source).name.removesuffix('_pet.nii.gz')
    reference = directory / f'{stem}_desc-hmc_petref.nii.gz'
    image = nb.load(source)
    data = image.get_fdata()
    nb.Nifti1Image(data[..., 0] if image.ndim == 4 else data, image.affine).to_filename(reference)
    # Both historical (no desc) and current (desc-hmc) names must be discoverable.
    transform = directory / (
        f'{stem}_from-orig_to-petref_mode-image' + ('_desc-hmc' if desc else '') + '_xfm.txt'
    )
    matrices = np.tile(np.eye(4), (nframes, 1, 1))
    matrices[:, 0, 3] = np.arange(nframes)
    nt.linear.LinearTransformsMapping(matrices).to_filename(transform, fmt='itk')
    return {'hmc_petref': str(reference), 'transforms': {'hmc': str(transform)}}


@pytest.fixture
def pet_only_config(tmp_path, minimal_bids):
    (minimal_bids / 'sub-01/anat/sub-01_T1w.nii.gz').unlink()
    try:
        parse_args(
            [
                str(minimal_bids),
                str(tmp_path / 'out'),
                'participant',
                '--pet-only',
                '--output-spaces',
                'MNI152NLin2009cAsym:res-2',
                '--seg',
                'Schaefer2018100Parcels7Networks',
                '--skip-bids-validation',
                '--notrack',
                '-w',
                str(tmp_path / 'work'),
                '--nthreads',
                '1',
            ]
        )
        config.init_spaces()
        yield minimal_bids
    finally:
        _reset_config()


@pytest.mark.parametrize('nframes', [1, 3])
def test_pet_only_graph(pet_only_config, nframes):
    pet = pet_only_config / 'sub-01/pet/sub-01_pet.nii.gz'
    shape = (12, 12, 12) if nframes == 1 else (12, 12, 12, nframes)
    nb.Nifti1Image(np.ones(shape), np.eye(4)).to_filename(pet)
    pet.with_suffix('').with_suffix('.json').write_text(
        json.dumps(
            {
                'FrameTimesStart': list(range(nframes)),
                'FrameDuration': [1] * nframes,
            }
        )
    )
    assert config.workflow.pet_only
    assert config.workflow.petref == 'twa'
    assert not config.workflow.run_reconall
    assert config.execution.participant_label == ['01']
    assert config.workflow.spaces.get_spaces() == ['MNI152NLin2009cAsym']
    workflow = init_petprep_wf()
    nodes = workflow.list_node_names()
    assert not any('anat_fit' in node or 'reconall' in node for node in nodes)
    assert any('pet_template_reg_wf.register' in node for node in nodes)
    assert any('ds_morph' in node for node in nodes)
    assert any('pet_hmc_wf' in node for node in nodes) == (nframes > 1)
    workflow._create_flat_graph()


@pytest.mark.parametrize('desc', [False, True])
def test_pet_only_reuses_discovered_motion(pet_only_config, tmp_path, desc):
    """--derivatives must skip HMC even with first5min and an empty work directory."""
    pet = pet_only_config / 'sub-01/pet/sub-01_pet.nii.gz'
    nb.Nifti1Image(np.ones((12, 12, 12, 3)), np.eye(4)).to_filename(pet)
    derivatives = tmp_path / 'previous'
    cache = _write_motion_cache(derivatives / 'sub-01/pet', pet, desc=desc)
    config.execution.derivatives = {'petprep': derivatives}
    config.workflow.petref = 'first5min'
    config.workflow.petref_specified = True
    workflow = init_petprep_wf()
    nodes = workflow._get_all_nodes()
    assert not any('pet_hmc_wf' in name for name in workflow.list_node_names())
    assert not any(
        node.interface.__class__.__module__.startswith('nipype.interfaces.freesurfer')
        for node in nodes
    )
    buffer = next(node for node in nodes if node.name == 'motion_buffer')
    assert buffer.inputs.motion_reference == cache['hmc_petref']
    assert buffer.inputs.motion_xfm == cache['transforms']['hmc']
    assert next(node for node in nodes if node.name == 'reference').inputs.strategy == 'first5min'
    assert any('pet_template_reg_wf.register' in name for name in workflow.list_node_names())
    workflow._create_flat_graph()


@pytest.mark.parametrize('strategy', ['first5min', 'template'])
def test_cached_motion_reference_execution(pet_only_config, tmp_path, monkeypatch, strategy):
    """Apply cached nonidentity transforms, then build the currently requested reference."""
    from petprep.interfaces.resampling import ResampleSeries
    from petprep.utils.pet_only_reuse import find_reference
    from petprep.workflows.pet.pet_only import init_pet_only_prepare_wf

    monkeypatch.chdir(tmp_path)
    pet = pet_only_config / 'sub-01/pet/sub-01_pet.nii.gz'
    data = np.zeros((10, 10, 10, 3), dtype='float32')
    data[2:7, 2:7, 2:7, :] = [2, 6, 20]
    nb.Nifti1Image(data, np.eye(4)).to_filename(pet)
    cache = _write_motion_cache(tmp_path / 'previous/sub-01/pet', pet)
    original = {
        path: Path(path).read_bytes() for path in (cache['hmc_petref'], cache['transforms']['hmc'])
    }
    config.workflow.petref = strategy
    metadata = {'FrameTimesStart': [0, 100, 300], 'FrameDuration': [100, 200, 100]}
    workflow = init_pet_only_prepare_wf(pet_file=str(pet), metadata=metadata, precomputed=cache)
    workflow.base_dir = str(tmp_path / 'fresh-work')
    assert not any('pet_hmc_wf' in name for name in workflow.list_node_names())
    assert ('pet_reference_wf.motion_corrected' in workflow.list_node_names()) == (
        strategy != 'template'
    )
    workflow.run(plugin='Linear')
    saved = next(config.execution.petprep_dir.rglob(f'*desc-{strategy}_petref.nii.gz'))
    if strategy == 'template':
        expected = nb.load(cache['hmc_petref']).get_fdata()
    else:
        corrected = (
            ResampleSeries(
                in_file=str(pet),
                ref_file=cache['hmc_petref'],
                transforms=[cache['transforms']['hmc']],
                mode='constant',
            )
            .run(cwd=str(tmp_path))
            .outputs.out_file
        )
        expected = nb.load(prepare_reference(corrected, cache['hmc_petref'], metadata, strategy))
        expected = expected.get_fdata()
        uncorrected = nb.load(prepare_reference(str(pet), cache['hmc_petref'], metadata, strategy))
        assert not np.allclose(expected, uncorrected.get_fdata())
    np.testing.assert_allclose(nb.load(saved).get_fdata(), expected)
    assert all(Path(path).read_bytes() == data for path, data in original.items())
    assert not list(config.execution.petprep_dir.rglob('*_xfm.txt'))

    recovered = find_reference(
        str(pet), metadata, cache, {'petprep': config.execution.petprep_dir}, strategy, False
    )
    assert recovered == str(saved)
    reused = init_pet_only_prepare_wf(
        pet_file=str(pet), metadata=metadata, precomputed=cache, reference_cache=recovered
    )
    assert not any('pet_hmc_wf' in name for name in reused.list_node_names())
    assert not any(name.endswith('.reference') for name in reused.list_node_names())
    assert not any(isinstance(node.interface, ResampleSeries) for node in reused._get_all_nodes())


def test_motion_outputs_can_be_rediscovered(pet_only_config, tmp_path, monkeypatch):
    """A new PET-only HMC fit persists the native pair for reuse in a later run."""
    from nipype.interfaces import utility as niu
    from nipype.pipeline import engine as pe

    from petprep.utils.pet_only import collect_pet_only_derivatives
    from petprep.workflows.pet import pet_only

    pet = pet_only_config / 'sub-01/pet/sub-01_pet.nii.gz'
    nb.Nifti1Image(np.ones((12, 12, 12, 3)), np.eye(4)).to_filename(pet)
    cache = _write_motion_cache(tmp_path / 'estimated', pet)

    def estimated_hmc(**kwargs):
        workflow = pe.Workflow(name='pet_hmc_wf')
        inputnode = pe.Node(niu.IdentityInterface(fields=['pet_file']), name='inputnode')
        outputnode = pe.Node(niu.IdentityInterface(fields=['petref', 'xforms']), name='outputnode')
        outputnode.inputs.petref = cache['hmc_petref']
        outputnode.inputs.xforms = cache['transforms']['hmc']
        workflow.add_nodes([inputnode, outputnode])
        return workflow

    monkeypatch.setattr(pet_only, 'init_pet_hmc_wf', estimated_hmc)
    workflow = pet_only.init_pet_only_hmc_wf(pet_file=str(pet), metadata={})
    workflow.base_dir = str(tmp_path / 'first-work')
    workflow.run(plugin='Linear')
    recovered = collect_pet_only_derivatives(str(pet), {'petprep': config.execution.petprep_dir})
    assert recovered['hmc_petref'].endswith('_desc-hmc_petref.nii.gz')
    assert (
        Path(recovered['transforms']['hmc']).read_bytes()
        == Path(cache['transforms']['hmc']).read_bytes()
    )
    rerun = pet_only.init_pet_only_hmc_wf(pet_file=str(pet), metadata={}, precomputed=recovered)
    assert not any('pet_hmc_wf' in name for name in rerun.list_node_names())


@pytest.mark.parametrize('missing', ['hmc_petref', 'transforms'])
def test_incomplete_motion_cache(pet_only_config, tmp_path, missing):
    from petprep.workflows.pet.pet_only import init_pet_only_prepare_wf

    pet = pet_only_config / 'sub-01/pet/sub-01_pet.nii.gz'
    nb.Nifti1Image(np.ones((12, 12, 12, 3)), np.eye(4)).to_filename(pet)
    cache = _write_motion_cache(tmp_path / 'previous', pet)
    cache.pop(missing)
    with pytest.raises(ValueError, match='requires both desc-hmc_petref'):
        init_pet_only_prepare_wf(pet_file=str(pet), metadata={}, precomputed=cache)
    config.workflow.hmc_off = True
    workflow = init_pet_only_prepare_wf(pet_file=str(pet), metadata={}, precomputed=cache)
    assert any('identity_motion' in name for name in workflow.list_node_names())
    assert not any('pet_hmc_wf' in name for name in workflow.list_node_names())


def test_wrong_motion_frame_count(tmp_path):
    from petprep.utils.pet_only import select_motion_derivatives

    pet = tmp_path / 'sub-01_pet.nii.gz'
    nb.Nifti1Image(np.ones((5, 5, 5, 3)), np.eye(4)).to_filename(pet)
    cache = _write_motion_cache(tmp_path / 'previous', pet, nframes=2)
    with pytest.raises(ValueError, match=r'count \(2\).*frames \(3\)'):
        select_motion_derivatives(cache, 3)


def test_motion_cache_matches_acquisition_and_keeps_pair(tmp_path):
    from petprep.utils.pet_only import collect_pet_only_derivatives

    source = tmp_path / 'sub-01_ses-A_trc-FDG_rec-ac_run-1_pet.nii.gz'
    nb.Nifti1Image(np.ones((5, 5, 5, 3)), np.eye(4)).to_filename(source)
    first, second = tmp_path / 'first', tmp_path / 'second'
    cache = _write_motion_cache(first / 'sub-01/ses-A/pet', source)
    other = tmp_path / 'sub-01_ses-A_trc-FDG_rec-ac_run-2_pet.nii.gz'
    shutil.copyfile(source, other)
    _write_motion_cache(first / 'sub-01/ses-A/pet', other)
    partial = _write_motion_cache(second / 'sub-01/ses-A/pet', source)
    Path(partial['hmc_petref']).unlink()
    assert collect_pet_only_derivatives(str(source), {'a': first, 'b': second}) == cache
    no_run = tmp_path / 'sub-01_ses-A_trc-FDG_rec-ac_pet.nii.gz'
    shutil.copyfile(source, no_run)
    assert collect_pet_only_derivatives(str(no_run), {'a': first}) == {}


@pytest.mark.parametrize(
    'options',
    [
        ['--anat-only'],
        ['--output-spaces', 'T1w'],
        ['--petref', 'auto'],
    ],
)
def test_pet_only_rejects_anatomical_requests(tmp_path, minimal_bids, options):
    try:
        with pytest.raises(SystemExit):
            parse_args(
                [
                    str(minimal_bids),
                    str(tmp_path / 'out'),
                    'participant',
                    '--pet-only',
                    '--skip-bids-validation',
                    '--notrack',
                    *options,
                ]
            )
    finally:
        _reset_config()


@pytest.mark.parametrize(
    ('spaces', 'expected', 'resolution'),
    [
        ([], ['MNI152NLin2009cAsym'], '1'),
        (['--output-spaces', 'MNI152NLin6Asym:res-2'], ['MNI152NLin6Asym'], '2'),
        (
            ['--output-spaces', 'MNI152NLin2009cAsym:res-native'],
            ['MNI152NLin2009cAsym'],
            'native',
        ),
    ],
)
def test_pet_only_default_and_explicit_spaces(
    tmp_path, minimal_bids, spaces, expected, resolution
):
    try:
        parse_args(
            [
                str(minimal_bids),
                str(tmp_path / 'out'),
                'participant',
                '--pet-only',
                '--skip-bids-validation',
                '--notrack',
                '--seg',
                'Schaefer2018100Parcels7Networks',
                '-w',
                str(tmp_path / 'work'),
                *spaces,
            ]
        )
        config.init_spaces()
        assert config.workflow.spaces.get_spaces() == expected
        reference = config.workflow.spaces.references[0]
        assert str(reference.spec['res']) == resolution
        if not spaces:
            _check_default_extraction_grid(tmp_path, reference)
    finally:
        _reset_config()


def _check_default_extraction_grid(tmp_path, reference):
    """Resample 2 mm PET, then measure atlas volumes and TACs on the default grid."""
    import nitransforms as nt
    from nipype.interfaces.utility import Function

    from petprep.workflows.pet.apply import init_pet_volumetric_resample_wf

    pet = tmp_path / 'coarse_pet.nii.gz'
    fixed = tmp_path / 'template.nii.gz'
    seg = tmp_path / 'atlas.nii.gz'
    table = tmp_path / 'labels.tsv'
    metadata = tmp_path / 'metadata.json'
    identity = tmp_path / 'identity.tfm'
    values = (10 + 2 * np.indices((9, 9, 9))[0]).astype('float32')
    nb.Nifti1Image(values, np.diag([2.0, 2.0, 2.0, 1.0])).to_filename(pet)
    nb.Nifti1Image(np.ones((17, 17, 17), dtype='uint8'), np.eye(4)).to_filename(fixed)
    labels = np.zeros((17, 17, 17), dtype='int16')
    labels[6:11, 6:11, 6:11] = 1
    nb.Nifti1Image(labels, np.eye(4)).to_filename(seg)
    table.write_text('index\tname\n1\tregion\n')
    metadata.write_text(json.dumps({'FrameTimesStart': [0], 'FrameDuration': [300]}))
    nt.Affine().to_filename(identity, fmt='itk')
    sampler = init_pet_volumetric_resample_wf(mem_gb={'resampled': 0.01}, direct=True)
    sampler.base_dir = str(tmp_path / 'sampling')
    sampler.config['execution']['crashdump_dir'] = str(tmp_path)
    sampler.inputs.inputnode.pet_file = str(pet)
    sampler.inputs.inputnode.pet_ref_file = str(pet)
    sampler.inputs.inputnode.target_ref_file = str(fixed)
    sampler.inputs.inputnode.target_mask = str(fixed)
    sampler.inputs.inputnode.motion_xfm = str(identity)
    sampler.inputs.inputnode.petref2target_xfm = str(identity)
    sampler.inputs.inputnode.resolution = reference.spec['res']
    graph = sampler.run(plugin='Linear')
    sampled = next(node for node in graph.nodes if node.name == 'resample').result.outputs.out_file
    image = nb.load(sampled)
    assert image.shape == (17, 17, 17)
    np.testing.assert_array_equal(image.header.get_zooms(), (1, 1, 1))
    np.testing.assert_array_equal(image.affine, np.eye(4))
    sample_atlas = Function(
        function=template_segmentation,
        input_names=['segmentation', 'reference', 'label_table', 'space'],
        output_names=['segmentation', 'morph', 'metadata'],
        imports=['from pathlib import Path', 'import nibabel as nb', 'import numpy as np'],
    )
    sample_atlas.inputs.segmentation = str(seg)
    sample_atlas.inputs.reference = sampled
    sample_atlas.inputs.label_table = str(table)
    sample_atlas.inputs.space = reference.space
    atlas_result = sample_atlas.run(cwd=str(tmp_path)).outputs
    assert pd.read_csv(atlas_result.morph, sep='\t')['volume-mm3'].iloc[0] == 125
    curves = ExtractTACs(
        in_file=sampled,
        segmentation=atlas_result.segmentation,
        dseg_tsv=str(table),
        metadata=str(metadata),
    ).run(cwd=str(tmp_path))
    np.testing.assert_allclose(pd.read_csv(curves.outputs.out_file, sep='\t')['region'], [18])


@pytest.mark.parametrize(
    ('suffix', 'extension'),
    [('dseg', '.tsv'), ('morph', '.tsv'), ('tacs', '.tsv'), ('xfm', '.h5')],
)
def test_derivative_resolutions_do_not_collide(tmp_path, suffix, extension):
    from petprep.interfaces.pet_only import PETOnlyDataSink

    source = tmp_path / 'sub-01_ses-02_task-rest_trc-FDG_rec-ac_run-1_pet.nii.gz'
    nb.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)).to_filename(source)
    input_file = tmp_path / f'input{extension}'
    input_file.write_text('test')
    paths = []
    for resolution in (1, 2):
        kwargs = (
            {'from': 'petref', 'to': 'Example', 'mode': 'image'}
            if suffix == 'xfm'
            else {'space': 'Example', 'seg': 'Test', 'allowed_entities': ('seg',)}
        )
        result = PETOnlyDataSink(
            source_file=str(source),
            in_file=str(input_file),
            base_directory=str(tmp_path / 'out'),
            suffix=suffix,
            datatype='pet',
            resolution=resolution,
            extension=extension,
            check_hdr=False,
            **kwargs,
        ).run(cwd=str(tmp_path))
        path = Path(result.outputs.out_file)
        assert f'res-{resolution}' in path.name
        assert 'task-rest_trc-FDG_rec-ac_run-1' in path.name
        assert path.parent.name == 'pet'
        assert path.parent.parent.name == 'ses-02'
        paths.append(path)
    assert paths[0] != paths[1]


def test_static_reference_is_3d(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(np.ones((5, 5, 5, 1)), np.eye(4)).to_filename(path)
    reference = prepare_reference(str(path), str(path), {}, 'twa')
    assert nb.load(reference).shape == (5, 5, 5)


def test_template_labels_morph_and_tacs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    grid = np.diag([2.0, 2.0, 2.0, 1.0])
    seg = np.zeros((6, 6, 6), dtype='int16')
    seg[1:3, 1:5, 1:5] = 1
    seg[3:5, 1:5, 1:5] = 2
    values = np.zeros((*seg.shape, 2), dtype='float32')
    values[seg == 1] = [2, 4]
    values[seg == 2] = [5, 10]
    for name, data in [('seg', seg), ('pet', values)]:
        nb.Nifti1Image(data, grid).to_filename(tmp_path / f'{name}.nii.gz')
    table = tmp_path / 'labels.tsv'
    table.write_text('index\tname\n1\tone\n2\ttwo\n3\tabsent\n')
    aligned, morph, meta = template_segmentation(
        str(tmp_path / 'seg.nii.gz'), str(tmp_path / 'pet.nii.gz'), str(table), 'Example'
    )
    assert meta['SubjectSpecific'] is False
    np.testing.assert_allclose(pd.read_csv(morph, sep='\t')['volume-mm3'], [256, 256, 0])
    support = np.ones(values.shape)
    support[1, 1, 1, 1] = 0  # Partial coverage invalidates this region in frame 2 only.
    nb.Nifti1Image(support, grid).to_filename(tmp_path / 'support.nii.gz')
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'FrameTimesStart': [0, 10], 'FrameDuration': [10, 20]}))
    result = ExtractTACs(
        in_file=str(tmp_path / 'pet.nii.gz'),
        segmentation=aligned,
        dseg_tsv=str(table),
        metadata=str(metadata),
        support=str(tmp_path / 'support.nii.gz'),
    ).run(cwd=str(tmp_path))
    curves = pd.read_csv(result.outputs.out_file, sep='\t')
    assert curves['one'][0] == 2
    assert np.isnan(curves['one'][1])
    np.testing.assert_allclose(curves['two'], [5, 10])
    assert curves['absent'].isna().all()


def test_resource_queries_use_requested_space(tmp_path, monkeypatch):
    image = tmp_path / 'image.nii.gz'
    nb.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)).to_filename(image)
    table = tmp_path / 'labels.tsv'
    table.write_text('index\tname\n1\tone\n')
    queries = []

    def resolve(template, resource):
        queries.append((template, resource['query']))
        return str(table if resource['query']['extension'] == '.tsv' else image)

    monkeypatch.setattr(atlas, '_resolve_resource', resolve)
    result = resolve_resources('MNI152NLin2009cAsym', {'res': 2}, 'gtm')
    assert all(template == 'MNI152NLin2009cAsym' for template, _ in queries)
    assert queries[0][1]['suffix'] == 'T1w'
    assert queries[2][1]['atlas'] == 'gtm'
    assert 'resolution' not in queries[3][1]
    assert result[-1]['RegistrationContrast'] == 'T1w'


def test_hocpa_uses_shared_label_table_from_templateflow(tmp_path, monkeypatch):
    image = tmp_path / 'image.nii.gz'
    nb.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)).to_filename(image)
    table = tmp_path / 'labels.tsv'
    table.write_text('index\tname\n1\tone\n')

    def resolve(template, resource):
        assert resource['source'] == 'templateflow'
        if resource['query']['extension'] == '.tsv':
            # Match the reported failure: 2009c has an image but no HOCPA TSV.
            if template != 'MNI152NLin6Asym':
                raise ValueError('No files found for atlas resource')
            return str(table)
        assert template == 'MNI152NLin2009cAsym'
        return str(image)

    monkeypatch.setattr(atlas, '_resolve_resource', resolve)
    result = resolve_resources('MNI152NLin2009cAsym', {'res': 'native'}, 'HOCPA')
    assert result[2:4] == (str(image), str(table))
    provenance = result[-1]
    assert provenance['SegmentationSource'] == 'TemplateFlow'
    assert provenance['Resources']['segmentation']['Query']['template'] == 'MNI152NLin2009cAsym'
    assert provenance['Resources']['labels']['Query']['template'] == 'MNI152NLin6Asym'


@pytest.mark.parametrize('missing', ['segmentation', 'labels'])
def test_missing_templateflow_segmentation_is_actionable(tmp_path, monkeypatch, missing):
    image = tmp_path / 'image.nii.gz'
    nb.Nifti1Image(np.ones((5, 5, 5)), np.eye(4)).to_filename(image)

    def resolve(template, resource):
        query = resource['query']
        if query.get('atlas') == 'brainstem' and (
            missing == 'segmentation' or query['extension'] == '.tsv'
        ):
            raise ValueError('No files found for atlas resource')
        return str(image)

    monkeypatch.setattr(atlas, '_resolve_resource', resolve)
    with pytest.raises(ValueError, match=f'PET-only {missing} unavailable') as error:
        resolve_resources('MNI152NLin2009cAsym', {}, 'brainstem')
    assert 'TemplateFlow query:' in str(error.value)
    assert '--seg brainstem' in str(error.value)
    assert 'Install or publish' in str(error.value)


@pytest.mark.parametrize('segmentation', [*atlas.SUBJECT_SEGMENTATIONS, 'HOCPA', 'MASSP20'])
def test_pet_only_segmentation_never_runs_subject_estimation(pet_only_config, segmentation):
    config.workflow.seg = segmentation
    workflow = init_petprep_wf()
    nodes = workflow._get_all_nodes()
    assert (
        next(node for node in nodes if node.name == 'resources').inputs.segmentation
        == segmentation
    )
    assert not any(
        node.interface.__class__.__module__ == 'petprep.interfaces.segmentation' for node in nodes
    )
    assert not any('segmentation_wf' in name for name in workflow.list_node_names())
    assert config.workflow.spaces.get_spaces() == ['MNI152NLin2009cAsym']


@pytest.mark.parametrize(
    'linear',
    [
        np.eye(3),
        np.diag([1.0, 2.0, 3.0]),
        np.array([[0.0, -2.0, 0.0], [1.0, 0.0, 0.5], [0.0, 0.0, 3.0]]),
    ],
)
def test_registration_mask_physical_radius(tmp_path, monkeypatch, linear):
    """The mask extends 5 mm, including on anisotropic and non-axis-aligned grids."""
    monkeypatch.chdir(tmp_path)
    data = np.zeros((25, 25, 25), dtype='uint8')
    points = np.array([[12, 12, 12], [15, 16, 10]])
    data[tuple(points.T)] = 1
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = [-17, 8, 23]
    source = tmp_path / 'brain_mask.nii.gz'
    nb.Nifti1Image(data, affine).to_filename(source)
    original = source.read_bytes()

    masks = registration_masks(str(source), 5.0)
    assert masks[:2] == ['NULL', 'NULL']
    result = nb.load(masks[2])
    np.testing.assert_allclose(result.affine, affine)
    assert result.shape == data.shape
    assert result.get_data_dtype() == np.dtype('uint8')
    world = nb.affines.apply_affine(affine, np.indices(data.shape).reshape(3, -1).T)
    centers = nb.affines.apply_affine(affine, points)
    distance = np.min(np.linalg.norm(world[:, None, :] - centers, axis=-1), axis=1)
    expected = (distance <= 5.0 + 1e-8).reshape(data.shape)
    np.testing.assert_array_equal(result.get_fdata(), expected)
    assert source.read_bytes() == original


def test_registration_mask_rejects_empty_mask(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'brain_mask.nii.gz'
    nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4)).to_filename(source)
    with pytest.raises(ValueError, match='nonempty 3D'):
        registration_masks(str(source), 5.0)


def test_com_initialization_direction(tmp_path, monkeypatch):
    import nitransforms as nt

    monkeypatch.chdir(tmp_path)
    indices = np.indices((15, 17, 19))
    data = np.exp(
        -sum(
            ((indices[d] - c) / s) ** 2
            for d, c, s in zip(range(3), (7, 8, 9), (2, 3, 4), strict=True)
        )
    )
    fixed = tmp_path / 'fixed.nii.gz'
    moving = tmp_path / 'moving.nii.gz'
    mask = tmp_path / 'mask.nii.gz'
    affine = np.eye(4)
    shift = [7, -4, 2]
    moved_affine = affine.copy()
    moved_affine[:3, 3] = shift
    nb.Nifti1Image(data, affine).to_filename(fixed)
    nb.Nifti1Image(data, moved_affine).to_filename(moving)
    nb.Nifti1Image(np.ones(data.shape), affine).to_filename(mask)
    files, names = registration_initializations(str(fixed), str(moving), str(mask))
    assert names == ['header', 'com', 'principalaxes']
    transform = nt.linear.load(files[1])
    np.testing.assert_allclose(transform.map([7, 8, 9])[0], np.array([7, 8, 9]) + shift)


@pytest.mark.skipif(shutil.which('antsRegistration') is None, reason='ANTs is required')
def test_real_ants_normalization(tmp_path):
    import nitransforms as nt
    from nipype.interfaces.ants import ApplyTransforms
    from scipy.ndimage import map_coordinates

    from petprep.interfaces.resampling import ResampleSeries

    coords = np.indices((32, 32, 32))
    data = np.exp(
        -sum(
            ((coords[d] - c) / s) ** 2
            for d, c, s in zip(range(3), (15, 16, 14), (5, 7, 9), strict=True)
        )
    )
    fixed = tmp_path / 'fixed.nii.gz'
    moving = tmp_path / 'moving.nii.gz'
    mask = tmp_path / 'mask.nii.gz'
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [30, -20, -15]
    deformed = coords.astype(float)
    deformed[0] += 1.5 * np.sin(coords[2] / 6)
    moving_data = map_coordinates(data, deformed, order=1)
    nb.Nifti1Image(data.astype('float32'), affine).to_filename(fixed)
    nb.Nifti1Image(moving_data.astype('float32'), affine).to_filename(moving)
    nb.Nifti1Image((data > 0.05).astype('uint8'), affine).to_filename(mask)
    workflow = init_pet_template_reg_wf(sloppy=True)
    workflow.base_dir = str(tmp_path)
    workflow.config['execution']['remove_unnecessary_outputs'] = False
    workflow.config['execution']['crashdump_dir'] = str(tmp_path)
    workflow.inputs.inputnode.petref = str(moving)
    workflow.inputs.inputnode.template = str(fixed)
    workflow.inputs.inputnode.template_mask = str(mask)
    graph = workflow.run(plugin='Linear')
    masks = next(node for node in graph.nodes if node.name == 'dilate_mask').result.outputs
    register = next(node for node in graph.nodes if node.name == 'register')
    for runtime in register.result.runtime:
        stages = runtime.cmdline.split('--transform ')[1:]
        assert len(stages) == 3
        assert '--masks [ NULL, NULL ]' in stages[0]
        assert '--masks [ NULL, NULL ]' in stages[1]
        assert f'--masks [ {masks.fixed_image_masks[2]}, NULL ]' in stages[2]
    selected = next(node for node in graph.nodes if node.name == 'select').result.outputs
    assert selected.forward.endswith('.h5')
    assert selected.inverse.endswith('.h5')
    record = json.loads(Path(selected.report).read_text())
    assert record['Candidates']
    assert np.isfinite(nb.load(selected.warped).get_fdata()).all()

    # Compare the composed nonlinear + distinct per-frame motion path with ANTs.
    # This exercises a nontrivial NIfTI orientation and real composite HDF5 files.
    matrices = [np.eye(4), np.eye(4)]
    matrices[1][:3, 3] = [1.2, -0.7, 0.4]
    hmc = tmp_path / 'hmc.txt'
    nt.linear.LinearTransformsMapping(matrices).to_filename(hmc, fmt='itk')
    series = tmp_path / 'series.nii.gz'
    nb.Nifti1Image(np.stack([moving_data, 2 * moving_data], axis=-1), affine).to_filename(series)
    resampler = ResampleSeries(
        in_file=str(series),
        ref_file=str(fixed),
        transforms=[str(hmc), selected.forward],
        mode='constant',
        order=1,
    )
    composed = nb.load(resampler.run(cwd=str(tmp_path)).outputs.out_file).get_fdata()
    for index, matrix in enumerate(matrices):
        frame = tmp_path / f'frame{index}.nii.gz'
        transform = tmp_path / f'frame{index}.tfm'
        nt.Affine(matrix).to_filename(transform, fmt='itk')
        nb.Nifti1Image((index + 1) * moving_data, affine).to_filename(frame)
        expected = ApplyTransforms(
            dimension=3,
            input_image=str(frame),
            reference_image=str(fixed),
            transforms=[selected.forward, str(transform)],
            interpolation='Linear',
            output_image=str(tmp_path / f'expected{index}.nii.gz'),
        ).run()
        ants_data = nb.load(expected.outputs.output_image).get_fdata()
        np.testing.assert_allclose(
            composed[4:-4, 4:-4, 4:-4, index], ants_data[4:-4, 4:-4, 4:-4], atol=2e-5
        )

    inverse_result = ResampleSeries(
        in_file=str(fixed),
        ref_file=str(moving),
        transforms=[selected.inverse],
        mode='constant',
        order=1,
    ).run(cwd=str(tmp_path))
    inverse_expected = ApplyTransforms(
        dimension=3,
        input_image=str(fixed),
        reference_image=str(moving),
        transforms=[selected.inverse],
        interpolation='Linear',
        output_image=str(tmp_path / 'inverse.nii.gz'),
    ).run()
    # The inverse affine can move even interior target voxels outside the source.
    # ITK and SciPy differ at that boundary (half-voxel support vs constant fill).
    # Compare where both sample inside the source, without relaxing the tolerance.
    world = nb.affines.apply_affine(affine, coords.reshape(3, -1).T)
    sampled = nb.affines.apply_affine(
        np.linalg.inv(affine), nt.manip.load(selected.inverse).map(world)
    )
    valid = np.all((sampled >= 1) & (sampled <= np.array(data.shape) - 2), axis=1)
    valid = valid.reshape(data.shape)
    interior = np.zeros(data.shape, dtype=bool)
    interior[4:-4, 4:-4, 4:-4] = True
    valid &= interior
    assert valid.any()
    np.testing.assert_allclose(
        nb.load(inverse_result.outputs.out_file).get_fdata()[valid],
        nb.load(inverse_expected.outputs.output_image).get_fdata()[valid],
        # The inverse affine samples the field off-grid. NiTransforms uses
        # cubic field interpolation, so unlike the forward grid-aligned test
        # above, allow 1% of the input peak for interpolation differences.
        atol=0.01 * data.max(),
    )


@pytest.mark.skipif(shutil.which('antsRegistration') is None, reason='ANTs is required')
def test_pet_only_derivatives(pet_only_config, tmp_path, monkeypatch):
    """Execute a complete static PET-only workflow with local synthetic resources."""
    coords = np.indices((32, 32, 32))
    data = np.exp(-sum(((coords[d] - 15) / s) ** 2 for d, s in enumerate((5, 7, 9))))
    data *= 2 + 0.5 * np.sin(coords[0]) + 0.4 * np.cos(coords[1] * 0.8)
    pet = pet_only_config / 'sub-01/pet/sub-01_pet.nii.gz'
    nb.Nifti1Image(data.astype('float32'), np.eye(4)).to_filename(pet)
    mask = tmp_path / 'mask.nii.gz'
    seg = tmp_path / 'atlas.nii.gz'
    table = tmp_path / 'labels.tsv'
    nb.Nifti1Image((data > 0.05).astype('uint8'), np.eye(4)).to_filename(mask)
    nb.Nifti1Image((data > 0.1).astype('uint8'), np.eye(4)).to_filename(seg)
    table.write_text('index\tname\n1\tregion\n')

    def resolve(template, resource):
        query = resource['query']
        if query['extension'] == '.tsv':
            return str(table)
        return str({'T1w': pet, 'mask': mask, 'dseg': seg}[query['suffix']])

    monkeypatch.setattr(atlas, '_resolve_resource', resolve)
    config.execution.sloppy = True
    workflow = init_petprep_wf()
    workflow.config['execution']['crashdump_dir'] = str(tmp_path)
    workflow.run(plugin='Linear')
    outputs = config.execution.petprep_dir
    normalized = list(outputs.rglob('*space-*_desc-preproc_pet.nii.gz'))
    assert len(normalized) == 1
    assert 'space-MNI152NLin2009cAsym' in normalized[0].name
    curves = pd.read_csv(next(outputs.rglob('*tacs.tsv')), sep='\t')
    image = nb.load(normalized[0]).get_fdata()
    labels = nb.load(next(outputs.rglob('*dseg.nii.gz'))).get_fdata()
    np.testing.assert_allclose(curves['region'], image[labels == 1].mean())
    assert json.loads(next(outputs.rglob('*desc-resources_pet.json')).read_text())['Resources']
    assert json.loads(next(outputs.rglob('*desc-registration_pet.json')).read_text())['Candidates']
    morph_json = json.loads(next(outputs.rglob('*morph.json')).read_text())
    assert morph_json['SubjectSpecific'] is False
    assert list(outputs.rglob('*from-petref_to-MNI152NLin2009cAsym*xfm.h5'))

    # A different atlas in a fresh work directory must reuse stages 2–4.
    from petprep.interfaces.resampling import ResampleSeries

    assert not list(outputs.rglob('*cache_*.json'))
    assert list(outputs.rglob('*desc-twa_petref.nii.gz'))
    forward = next(outputs.rglob('*from-petref_to-MNI152NLin2009cAsym_mode-image_xfm.h5'))
    transform_metadata = json.loads(forward.with_suffix('.json').read_text())
    assert 'sub-01_desc-twa_petref.nii.gz' in transform_metadata['Sources']
    assert transform_metadata['PETReferenceStrategy'] == 'twa'
    parameters = transform_metadata['RegistrationParameters']
    assert parameters['transforms'] == ['Rigid', 'Affine', 'SyN']
    assert parameters['transform_parameters'][-1] == [0.1, 3.0, 0.5]
    assert parameters['sigma_units'] == ['vox', 'vox', 'mm']
    assert parameters['number_of_iterations'][-1][-1] == 0
    assert transform_metadata['RegistrationMask'] == {
        'Source': mask.name,
        'DilationRadius': 5.0,
        'DilationRadiusUnits': 'mm',
        'DilationMethod': 'Euclidean',
        'Stages': ['SyN'],
    }
    normalized_metadata = json.loads(
        normalized[0].with_suffix('').with_suffix('.json').read_text()
    )
    assert normalized_metadata['RegistrationMask'] == transform_metadata['RegistrationMask']
    original_files = {
        path: path.read_bytes()
        for pattern in (
            '*desc-twa_petref.nii.gz',
            '*xfm.h5',
            '*desc-preproc_pet.nii.gz',
        )
        for path in outputs.rglob(pattern)
    }
    config.execution.derivatives = {'petprep': outputs}
    config.execution.work_dir = tmp_path / 'fresh-reuse-work'
    config.workflow.seg = 'HOCPA'
    reused = init_petprep_wf()
    names = reused.list_node_names()
    assert not any('motion_corrected' in name or name.endswith('.reference') for name in names)
    assert not any('pet_template_reg_wf.register' in name for name in names)
    assert not any(isinstance(node.interface, ResampleSeries) for node in reused._get_all_nodes())
    reused.config['execution']['crashdump_dir'] = str(tmp_path)
    reused.run(plugin='Linear')
    assert list(outputs.rglob('*seg-HOCPA*tacs.tsv'))
    assert all(path.read_bytes() == data for path, data in original_files.items())

    # Missing resampled support reruns only resampling; fitting remains reusable.
    support = next(outputs.rglob('*desc-support_probseg.nii.gz'))
    support_bytes = support.read_bytes()
    support.unlink()
    partial = init_petprep_wf()
    assert not any('pet_template_reg_wf.register' in name for name in partial.list_node_names())
    assert any(isinstance(node.interface, ResampleSeries) for node in partial._get_all_nodes())
    support.write_bytes(support_bytes)

    config.workflow.petref = 'first5min'
    changed_reference = init_petprep_wf().list_node_names()
    assert any('motion_corrected' in name for name in changed_reference)
    assert any('pet_template_reg_wf.register' in name for name in changed_reference)

    # Even if the requested reference already exists, transforms fitted with twa
    # cannot be reused for first5min. The labelled reference itself can be reused.
    native = outputs / 'sub-01/pet/sub-01_desc-twa_petref.nii.gz'
    first5min = native.with_name('sub-01_desc-first5min_petref.nii.gz')
    first5min.write_bytes(native.read_bytes())
    reference_settings = json.loads(native.with_name('sub-01_desc-twa_petref.json').read_text())
    reference_settings['PETReferenceStrategy'] = 'first5min'
    first5min.with_name('sub-01_desc-first5min_petref.json').write_text(
        json.dumps(reference_settings)
    )
    changed_reference = init_petprep_wf().list_node_names()
    assert not any('motion_corrected' in name for name in changed_reference)
    assert any('pet_template_reg_wf.register' in name for name in changed_reference)
    config.workflow.petref = 'twa'

    # Pre-mask registrations must be refitted, while the native reference survives.
    old_metadata = dict(transform_metadata)
    old_metadata.pop('RegistrationMask')
    forward.with_suffix('.json').write_text(json.dumps(old_metadata))
    changed_mask = init_petprep_wf().list_node_names()
    assert not any(
        'motion_corrected' in name or name.endswith('.reference') for name in changed_mask
    )
    assert any('pet_template_reg_wf.register' in name for name in changed_mask)
    forward.with_suffix('.json').write_text(json.dumps(transform_metadata))

    config.execution.sloppy = False
    changed_recipe = init_petprep_wf().list_node_names()
    assert not any('motion_corrected' in name for name in changed_recipe)
    assert any('pet_template_reg_wf.register' in name for name in changed_recipe)
    config.execution.sloppy = True

    from niworkflows.utils.spaces import Reference, SpatialReferences

    config.workflow.spaces = SpatialReferences(
        [
            Reference('MNI152NLin2009cAsym', {'res': 2}),
            Reference('MNI152NLin2009cAsym', {'res': 1}),
            Reference('MNI152NLin6Asym', {'res': 2}),
        ]
    )
    changed_space = init_petprep_wf().list_node_names()
    assert not any('target_0_wf.pet_template_reg_wf.register' in name for name in changed_space)
    assert any('target_1_wf.pet_template_reg_wf.register' in name for name in changed_space)
    assert any('target_2_wf.pet_template_reg_wf.register' in name for name in changed_space)
    assert not any('motion_corrected' in name for name in changed_space)
