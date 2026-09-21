"""PET-only reuse through labelled references and ordinary derivative sidecars."""

import json

import nibabel as nb
import numpy as np
import pytest

from petprep.utils.pet_only_reuse import (
    find_reference,
    find_target_derivatives,
    reference_metadata,
    registration_metadata,
    resampling_metadata,
)


def _write_derivative(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith('.nii.gz'):
        nb.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)).to_filename(path)
    else:
        path.write_bytes(b'transform')
    path.with_name(path.name.split('.', 1)[0] + '.json').write_text(json.dumps(metadata))
    return path


@pytest.fixture
def derivatives(tmp_path, monkeypatch):
    from petprep.utils import atlas

    root = tmp_path / 'derivatives'
    directory = root / 'sub-01/ses-A/pet'
    source = tmp_path / 'sub-01_ses-A_trc-H2O_run-1_pet.nii.gz'
    nb.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)).to_filename(source)
    metadata = {'FrameTimesStart': [0], 'FrameDuration': [300], 'Units': 'Bq/mL'}
    stem = source.name.removesuffix('_pet.nii.gz')
    settings = reference_metadata(metadata, 'first5min', False)
    petref = _write_derivative(directory / f'{stem}_desc-first5min_petref.nii.gz', settings)
    fixed = tmp_path / 'tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz'
    mask = tmp_path / 'tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'
    monkeypatch.setattr(
        atlas,
        '_resolve_resource',
        lambda template, resource: str(fixed if resource['query']['suffix'] == 'T1w' else mask),
    )
    return source, metadata, petref, {'petprep': root}, fixed, mask


def test_reference_filename_and_settings(derivatives):
    source, metadata, petref, roots, _, _ = derivatives
    assert find_reference(source, metadata, {}, roots, 'first5min', False) == str(petref)
    assert find_reference(source, metadata, {}, roots, 'twa', False) is None
    assert find_reference(source, metadata, {}, roots, 'first5min', True) is None
    assert (
        find_reference(source, {**metadata, 'FrameDuration': [200]}, {}, roots, 'first5min', False)
        is None
    )
    # Older unlabelled references cannot identify the requested reference strategy.
    petref.rename(petref.with_name(petref.name.replace('first5min', 'registration')))
    assert find_reference(source, metadata, {}, roots, 'first5min', False) is None


@pytest.mark.parametrize('entity', ['run-1', 'trc-H2O', 'ses-A'])
def test_reference_requires_same_acquisition(derivatives, entity):
    source, metadata, _, roots, _, _ = derivatives
    other = source.with_name(source.name.replace(f'_{entity}', ''))
    other.write_bytes(source.read_bytes())
    assert find_reference(other, metadata, {}, roots, 'first5min', False) is None


def test_dynamic_reference_requires_reusable_motion(derivatives):
    source, metadata, _, roots, _, _ = derivatives
    nb.Nifti1Image(np.ones((4, 4, 4, 3)), np.eye(4)).to_filename(source)
    assert find_reference(source, metadata, {}, roots, 'first5min', False) is None


def test_template_space_reference_is_not_a_native_reference(derivatives):
    source, metadata, petref, roots, _, _ = derivatives
    warped = petref.with_name(petref.name.replace('_desc-', '_space-MNI152NLin2009cAsym_desc-'))
    _write_derivative(warped, reference_metadata(metadata, 'first5min', False))
    petref.unlink()
    assert find_reference(source, metadata, {}, roots, 'first5min', False) is None


@pytest.mark.parametrize(('res', 'cohort'), [('2', None), ('native', None), ('1', '1')])
def test_target_sources_and_partial_reuse(derivatives, res, cohort):
    source, metadata, petref, roots, fixed, mask = derivatives
    space = 'MNI152NLin2009cAsym' + (f'+{cohort}' if cohort else '')
    spec = {'res': res, **({'cohort': cohort} if cohort else {})}
    stem = source.name.removesuffix('_pet.nii.gz')
    directory = petref.parent
    regmeta = registration_metadata(
        petref, fixed, mask, reference_metadata(metadata, 'first5min', False), space, res, False
    )
    forward = _write_derivative(
        directory / f'{stem}_res-{res}_from-petref_to-{space}_mode-image_xfm.h5', regmeta
    )
    inverse = _write_derivative(
        directory / f'{stem}_res-{res}_from-{space}_to-petref_mode-image_xfm.h5', regmeta
    )
    spatial = f'_space-{space}' + (f'_res-{res}' if res != 'native' else '')
    warped = _write_derivative(
        directory / f'{stem}{spatial}_desc-first5min_petref.nii.gz', regmeta
    )

    def find(**changes):
        args = {
            'pet_file': source,
            'metadata': metadata,
            'petref': str(petref),
            'derivatives': roots,
            'template': 'MNI152NLin2009cAsym',
            'spec': spec,
            'strategy': 'first5min',
            'hmc_off': False,
            'sloppy': False,
        }
        return find_target_derivatives(**{**args, **changes})

    registration, resampling = find()
    assert registration == {
        'forward': str(forward),
        'inverse': str(inverse),
        'warped': str(warped),
    }
    assert resampling is None
    sampled_meta = resampling_metadata(source, forward, regmeta, metadata)
    pet = _write_derivative(directory / f'{stem}{spatial}_desc-preproc_pet.nii.gz', sampled_meta)
    support = _write_derivative(
        directory / f'{stem}{spatial}_desc-support_probseg.nii.gz', sampled_meta
    )
    assert find() == (registration, {'pet': str(pet), 'support': str(support)})
    assert find(petref=None) == (None, None)
    assert find(sloppy=True) == (None, None)
    assert find(template='MNI152NLin6Asym') == (None, None)
    assert find(spec={**spec, 'res': '3'}) == (None, None)
    if res in ('native', '2'):
        # The new 1 mm default must not select previous native/2 mm outputs.
        assert find(spec={**spec, 'res': '1'}) == (None, None)
    assert find(metadata={**metadata, 'Units': 'kBq/mL'}) == (registration, None)

    # Each difference from the previous default recipe prevents reuse of the
    # old registration and resampling, but the native reference remains valid.
    for key, value in (
        ('transform_parameters', [0.1, 3.0, 0.0]),
        ('sigma_units', 'vox'),
        ('number_of_iterations', [200, 100, 50, 25]),
    ):
        previous = json.loads(json.dumps(regmeta))
        previous['RegistrationParameters'][key][-1] = value
        _write_derivative(forward, previous)
        assert find() == (None, None)
        assert find_reference(source, metadata, {}, roots, 'first5min', False) == str(petref)
    _write_derivative(forward, regmeta)

    # Unmasked or differently masked registrations cannot be reused.
    for settings in (
        None,
        {**regmeta['RegistrationMask'], 'DilationRadius': 3.0},
        {**regmeta['RegistrationMask'], 'Stages': ['Affine', 'SyN']},
        {**regmeta['RegistrationMask'], 'Source': 'other_mask.nii.gz'},
    ):
        previous = dict(regmeta)
        if settings is None:
            previous.pop('RegistrationMask')
        else:
            previous['RegistrationMask'] = settings
        _write_derivative(forward, previous)
        assert find() == (None, None)
        assert find_reference(source, metadata, {}, roots, 'first5min', False) == str(petref)
    _write_derivative(forward, regmeta)

    # Resampling must also carry the matching mask provenance.
    stale = dict(sampled_meta)
    stale.pop('RegistrationMask')
    _write_derivative(pet, stale)
    assert find() == (registration, None)
    _write_derivative(pet, sampled_meta)

    # The reference filename itself participates in transform provenance.
    other_ref = petref.with_name(petref.name.replace('first5min', 'twa'))
    assert find(petref=str(other_ref)) == (None, None)
    # An older resampling made with another reference must not survive a new fit.
    stale = {**sampled_meta, 'RegistrationSources': [other_ref.name, fixed.name, mask.name]}
    _write_derivative(pet, stale)
    assert find() == (registration, None)
    _write_derivative(pet, sampled_meta)
    support.unlink()
    assert find() == (registration, None)
    inverse.unlink()
    assert find() == (None, None)


@pytest.mark.parametrize('change', ['missing', 'malformed', 'wrong_sources'])
def test_invalid_transform_sidecar(derivatives, change):
    source, metadata, petref, roots, fixed, mask = derivatives
    stem = source.name.removesuffix('_pet.nii.gz')
    space = 'MNI152NLin2009cAsym'
    expected = registration_metadata(
        petref, fixed, mask, reference_metadata(metadata, 'first5min', False), space, 2, False
    )
    forward = _write_derivative(
        petref.parent / f'{stem}_res-2_from-petref_to-{space}_mode-image_xfm.h5', expected
    )
    _write_derivative(
        petref.parent / f'{stem}_res-2_from-{space}_to-petref_mode-image_xfm.h5', expected
    )
    _write_derivative(
        petref.parent / f'{stem}_space-{space}_res-2_desc-first5min_petref.nii.gz', expected
    )
    sidecar = forward.with_suffix('.json')
    if change == 'missing':
        sidecar.unlink()
    elif change == 'malformed':
        sidecar.write_text('{')
    else:
        sidecar.write_text(json.dumps({**expected, 'Sources': ['another_petref.nii.gz']}))
    assert find_target_derivatives(
        source, metadata, str(petref), roots, space, {'res': 2}, 'first5min', False, False
    ) == (None, None)
