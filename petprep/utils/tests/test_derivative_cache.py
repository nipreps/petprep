from pathlib import Path

import pytest

from petprep.utils import bids


@pytest.mark.parametrize('desc', ['hmc', 'coreg'])
def test_baseline_found_as_str(tmp_path: Path, desc: str):
    subject = '01'

    to_find = tmp_path.joinpath(
        f'sub-{subject}', 'pet', f'sub-{subject}_desc-{desc}_petref.nii.gz'
    )
    to_find.parent.mkdir(parents=True)
    to_find.touch()

    entities = {
        'subject': subject,
        'datatype': 'pet',
        'suffix': 'petref',
        'extension': '.nii.gz',
    }

    derivs = bids.collect_derivatives(derivatives_dir=tmp_path, entities=entities)

    assert dict(derivs) == {
        'petref': str(to_find),
        f'{desc}_petref': str(to_find),
        'transforms': {},
    }


@pytest.mark.parametrize('xfm', ['petref2anat', 'hmc'])
def test_transforms_found_as_str(tmp_path: Path, xfm: str):
    subject = '01'
    fromto = {
        'hmc': 'from-orig_to-petref',
        'petref2anat': 'from-petref_to-anat',
    }[xfm]

    to_find = tmp_path.joinpath(
        f'sub-{subject}', 'pet', f'sub-{subject}_{fromto}_mode-image_xfm.txt'
    )
    to_find.parent.mkdir(parents=True)
    to_find.touch()

    entities = {
        'subject': subject,
        'suffix': 'pet',
        'extension': '.nii.gz',
    }

    derivs = bids.collect_derivatives(
        derivatives_dir=tmp_path,
        entities=entities,
    )
    assert derivs == {'transforms': {xfm: str(to_find)}}


def test_baseline_prefers_combined_run(tmp_path: Path):
    subject = '01'

    combined = tmp_path.joinpath(f'sub-{subject}', 'pet', f'sub-{subject}_desc-hmc_petref.nii.gz')
    run_file = tmp_path.joinpath(
        f'sub-{subject}', 'pet', f'sub-{subject}_run-1_desc-hmc_petref.nii.gz'
    )
    combined.parent.mkdir(parents=True)
    combined.touch()
    run_file.touch()

    entities = {
        'subject': subject,
        'datatype': 'pet',
        'suffix': 'petref',
        'extension': '.nii.gz',
    }

    from petprep import config

    combine_runs = getattr(config.workflow, 'combine_runs', False)
    config.workflow.combine_runs = True
    try:
        derivs = bids.collect_derivatives(derivatives_dir=tmp_path, entities=entities)
    finally:
        config.workflow.combine_runs = combine_runs

    assert dict(derivs) == {
        'petref': str(combined),
        'hmc_petref': str(combined),
        'transforms': {},
    }
