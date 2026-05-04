# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Test parser."""

from argparse import ArgumentError

import nibabel as nb
import numpy as np
import pytest
from packaging.version import Version

from ... import config
from ...tests.test_config import _reset_config
from .. import version as _version
from ..parser import _build_parser, parse_args

MIN_ARGS = ['data/', 'out/', 'participant']


@pytest.mark.parametrize(
    ('args', 'code'),
    [
        ([], 2),
        (MIN_ARGS, 2),  # bids_dir does not exist
        (MIN_ARGS + ['--fs-license-file'], 2),
        (MIN_ARGS + ['--fs-license-file', 'fslicense.txt'], 2),
    ],
)
def test_parser_errors(args, code):
    """Check behavior of the parser."""
    with pytest.raises(SystemExit) as error:
        _build_parser().parse_args(args)

    assert error.value.code == code


@pytest.mark.parametrize('args', [MIN_ARGS, MIN_ARGS + ['--fs-license-file']])
def test_parser_valid(tmp_path, args):
    """Check valid arguments."""
    datapath = tmp_path / 'data'
    datapath.mkdir(exist_ok=True)
    args[0] = str(datapath)

    if '--fs-license-file' in args:
        _fs_file = tmp_path / 'license.txt'
        _fs_file.write_text('')
        args.insert(args.index('--fs-license-file') + 1, str(_fs_file.absolute()))

    opts = _build_parser().parse_args(args)

    assert opts.bids_dir == datapath


@pytest.mark.parametrize(
    ('argval', 'gb'),
    [
        ('1G', 1),
        ('1GB', 1),
        ('1000', 1),  # Default units are MB
        ('32000', 32),  # Default units are MB
        ('4000', 4),  # Default units are MB
        ('1000M', 1),
        ('1000MB', 1),
        ('1T', 1000),
        ('1TB', 1000),
        ('1000000K', 1),
        ('1000000KB', 1),
        ('1000000000B', 1),
    ],
)
def test_memory_arg(tmp_path, argval, gb):
    """Check the correct parsing of the memory argument."""
    datapath = tmp_path / 'data'
    datapath.mkdir(exist_ok=True)
    _fs_file = tmp_path / 'license.txt'
    _fs_file.write_text('')

    args = [str(datapath)] + MIN_ARGS[1:] + ['--fs-license-file', str(_fs_file), '--mem', argval]
    opts = _build_parser().parse_args(args)

    assert opts.memory_gb == gb


@pytest.mark.parametrize(('current', 'latest'), [('1.0.0', '1.3.2'), ('1.3.2', '1.3.2')])
def test_get_parser_update(monkeypatch, capsys, current, latest):
    """Make sure the out-of-date banner is shown."""
    expectation = Version(current) < Version(latest)

    def _mock_check_latest(*args, **kwargs):
        return Version(latest)

    monkeypatch.setattr(config.environment, 'version', current)
    monkeypatch.setattr(_version, 'check_latest', _mock_check_latest)

    _build_parser()
    captured = capsys.readouterr().err

    msg = f"""\
You are using PETPrep-{current}, and a newer version of PETPrep is available: {latest}.
Please check out our documentation about how and when to upgrade:
https://petprep.readthedocs.io/en/latest/faq.html#upgrading"""

    assert (msg in captured) is expectation


@pytest.mark.parametrize('flagged', [(True, None), (True, 'random reason'), (False, None)])
def test_get_parser_blacklist(monkeypatch, capsys, flagged):
    """Make sure the blacklisting banner is shown."""

    def _mock_is_bl(*args, **kwargs):
        return flagged

    monkeypatch.setattr(_version, 'is_flagged', _mock_is_bl)

    _build_parser()
    captured = capsys.readouterr().err

    assert ('FLAGGED' in captured) is flagged[0]
    if flagged[0]:
        assert (flagged[1] or 'reason: unknown') in captured


def test_parse_args(tmp_path, minimal_bids):
    """Basic smoke test showing that our parse_args() function
    implements the BIDS App protocol"""
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'

    parse_args(
        args=[
            str(minimal_bids),
            str(out_dir),
            'participant',  # BIDS App
            '-w',
            str(work_dir),  # Don't pollute CWD
            '--skip-bids-validation',  # Empty files make BIDS sad
        ]
    )
    assert config.execution.layout.root == str(minimal_bids)
    _reset_config()


def test_parse_args_skips_subjects_missing_pet_or_t1w(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    img3d = nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
    img4d = nb.Nifti1Image(np.zeros((5, 5, 5, 1)), np.eye(4))

    t1w_01 = bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    t1w_01.parent.mkdir(parents=True, exist_ok=True)
    img3d.to_filename(t1w_01)
    pet_01 = bids / 'sub-01' / 'pet' / 'sub-01_pet.nii.gz'
    pet_01.parent.mkdir(parents=True, exist_ok=True)
    img4d.to_filename(pet_01)
    (pet_01.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    pet_02 = bids / 'sub-02' / 'pet' / 'sub-02_pet.nii.gz'
    pet_02.parent.mkdir(parents=True, exist_ok=True)
    img4d.to_filename(pet_02)
    (pet_02.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    t1w_03 = bids / 'sub-03' / 'anat' / 'sub-03_T1w.nii.gz'
    t1w_03.parent.mkdir(parents=True, exist_ok=True)
    img3d.to_filename(t1w_03)

    try:
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

        assert config.execution.participant_label == ['01']
    finally:
        _reset_config()


def test_parse_args_errors_when_all_subjects_missing_required_modalities(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    img3d = nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4))
    t1w_01 = bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    t1w_01.parent.mkdir(parents=True, exist_ok=True)
    img3d.to_filename(t1w_01)

    with pytest.raises(SystemExit):
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

    _reset_config()


def test_bids_filter_file(tmp_path, capsys):
    bids_path = tmp_path / 'data'
    out_path = tmp_path / 'out'
    bff = tmp_path / 'filter.json'
    args = [str(bids_path), str(out_path), 'participant', '--bids-filter-file', str(bff)]
    bids_path.mkdir()

    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(args)

    err = capsys.readouterr().err
    assert 'Path does not exist:' in err

    bff.write_text('{"invalid json": }')

    with pytest.raises(SystemExit):
        parser.parse_args(args)

    err = capsys.readouterr().err
    assert 'JSON syntax error in:' in err
    _reset_config()


def test_derivatives(tmp_path):
    """Check the correct parsing of the derivatives argument."""
    bids_path = tmp_path / 'data'
    out_path = tmp_path / 'out'
    args = [str(bids_path), str(out_path), 'participant']
    bids_path.mkdir()

    parser = _build_parser()

    # Providing --derivatives without a path should raise an error
    temp_args = args + ['--derivatives']
    with pytest.raises((SystemExit, ArgumentError)):
        parser.parse_args(temp_args)
    _reset_config()

    # Providing --derivatives without names should automatically label them
    temp_args = args + ['--derivatives', str(bids_path / 'derivatives/smriprep')]
    opts = parser.parse_args(temp_args)
    assert opts.derivatives == {'smriprep': bids_path / 'derivatives/smriprep'}
    _reset_config()

    # Providing --derivatives with names should use them
    temp_args = args + [
        '--derivatives',
        f'anat={str(bids_path / "derivatives/smriprep")}',
    ]
    opts = parser.parse_args(temp_args)
    assert opts.derivatives == {'anat': bids_path / 'derivatives/smriprep'}
    _reset_config()

    # Providing multiple unlabeled derivatives with the same name should raise an error
    temp_args = args + [
        '--derivatives',
        str(bids_path / 'derivatives_01/smriprep'),
        str(bids_path / 'derivatives_02/smriprep'),
    ]
    with pytest.raises(ValueError, match='Received duplicate derivative name'):
        parser.parse_args(temp_args)

    _reset_config()


def test_session_label_only_filters_pet(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    anat_path = bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    anat_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4)).to_filename(anat_path)

    pet_path = bids / 'sub-01' / 'ses-blocked' / 'pet' / 'sub-01_ses-blocked_pet.nii.gz'
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5, 1)), np.eye(4)).to_filename(pet_path)
    (pet_path.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    try:
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--session-label',
                'blocked',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

        filters = config.execution.bids_filters
        assert filters.get('pet', {}).get('session') == ['blocked']
        assert 'session' not in filters.get('anat', {})
    finally:
        _reset_config()


def test_tracer_label_only_filters_pet(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    anat_path = bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    anat_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4)).to_filename(anat_path)

    pet_path = bids / 'sub-01' / 'pet' / 'sub-01_trc-ucbj_pet.nii.gz'
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5, 1)), np.eye(4)).to_filename(pet_path)
    (pet_path.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    try:
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--tracer-label',
                'ucbj',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

        filters = config.execution.bids_filters
        assert filters.get('pet', {}).get('tracer') == ['ucbj']
        assert 'tracer' not in filters.get('anat', {})
    finally:
        _reset_config()


def test_tracer_label_validation(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    pet_path = bids / 'sub-01' / 'pet' / 'sub-01_trc-ucbj_pet.nii.gz'
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5, 1)), np.eye(4)).to_filename(pet_path)
    (pet_path.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    with pytest.raises(SystemExit):
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--tracer-label',
                'dasb',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

    _reset_config()


def test_run_label_only_filters_pet(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    anat_path = bids / 'sub-01' / 'anat' / 'sub-01_T1w.nii.gz'
    anat_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5)), np.eye(4)).to_filename(anat_path)

    pet_path = bids / 'sub-01' / 'pet' / 'sub-01_run-01_pet.nii.gz'
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5, 1)), np.eye(4)).to_filename(pet_path)
    (pet_path.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    try:
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--run-label',
                '01',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

        filters = config.execution.bids_filters
        assert filters.get('pet', {}).get('run') == [1]
        assert 'run' not in filters.get('anat', {})
    finally:
        _reset_config()


def test_run_label_validation(tmp_path):
    bids = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids.mkdir()
    (bids / 'dataset_description.json').write_text('{"Name": "Test", "BIDSVersion": "1.8.0"}')

    pet_path = bids / 'sub-01' / 'pet' / 'sub-01_run-01_pet.nii.gz'
    pet_path.parent.mkdir(parents=True, exist_ok=True)
    nb.Nifti1Image(np.zeros((5, 5, 5, 1)), np.eye(4)).to_filename(pet_path)
    (pet_path.with_suffix('').with_suffix('.json')).write_text(
        '{"FrameTimesStart": [0], "FrameDuration": [1]}'
    )

    with pytest.raises(SystemExit):
        parse_args(
            args=[
                str(bids),
                str(out_dir),
                'participant',
                '--run-label',
                '2',
                '--skip-bids-validation',
                '-w',
                str(work_dir),
            ]
        )

    _reset_config()


def test_pvc_argument_handling(tmp_path, minimal_bids):
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    base_args = [
        str(minimal_bids),
        str(out_dir),
        'participant',
        '-w',
        str(work_dir),
        '--skip-bids-validation',
    ]

    # Missing some PVC arguments should error
    with pytest.raises(SystemExit):
        parse_args(args=base_args + ['--pvc-tool', 'petpvc'])
    _reset_config()

    # Providing all PVC arguments should succeed and convert the PSF to a tuple
    parse_args(
        args=base_args
        + [
            '--pvc-tool',
            'petsurfer',
            '--pvc-method',
            'GTM',
            '--pvc-psf',
            '2',
            '2',
            '2',
        ]
    )
    assert config.workflow.pvc_tool == 'petsurfer'
    assert config.workflow.pvc_method == 'GTM'
    assert config.workflow.pvc_psf == (2.0, 2.0, 2.0)
    _reset_config()


def test_pvc_invalid_method(tmp_path, minimal_bids):
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    args = [
        str(minimal_bids),
        str(out_dir),
        'participant',
        '-w',
        str(work_dir),
        '--skip-bids-validation',
        '--pvc-tool',
        'petpvc',
        '--pvc-method',
        'BAD',
        '--pvc-psf',
        '5',
    ]

    with pytest.raises(SystemExit):
        parse_args(args=args)
    _reset_config()


def test_reference_mask_options(tmp_path, minimal_bids, monkeypatch, capsys):
    work_dir = tmp_path / 'work'
    base_args = [
        str(minimal_bids),
        str(tmp_path / 'out'),
        'participant',
        '-w',
        str(work_dir),
        '--skip-bids-validation',
    ]

    # Missing --ref-mask-name should raise error when --ref-mask-index is used
    with pytest.raises(SystemExit):
        parse_args(args=base_args + ['--ref-mask-index', '1', '2'])
    _reset_config()

    parse_args(args=base_args + ['--ref-mask-name', 'cerebellum', '--ref-mask-index', '3', '4'])
    assert config.workflow.ref_mask_name == 'cerebellum'
    assert config.workflow.ref_mask_index == (3, 4)
    _reset_config()

    # Default segmentation is GTM; semiovale is only defined for WM segmentation
    with pytest.raises(SystemExit):
        parse_args(args=base_args + ['--ref-mask-name', 'semiovale'])
    err = capsys.readouterr().err
    assert (
        "--ref-mask-name 'semiovale' is not available for --seg gtm, but only for --seg wm. "
        'Choose one of: cerebellum, neocortex, thalamus for --seg gtm.' in err
    )
    _reset_config()

    parse_args(args=base_args + ['--seg', 'wm', '--ref-mask-name', 'semiovale'])
    assert config.workflow.seg == 'wm'
    assert config.workflow.ref_mask_name == 'semiovale'
    _reset_config()


def test_reference_mask_validation_edge_cases(tmp_path, minimal_bids, monkeypatch, capsys):
    """Cover parser errors for unsupported masks with and without segmentation mappings."""
    from importlib import resources
    from importlib.resources import files as ir_files

    work_dir = tmp_path / 'work'
    base_args = [
        str(minimal_bids),
        str(tmp_path / 'out'),
        'participant',
        '-w',
        str(work_dir),
        '--skip-bids-validation',
    ]

    # Force a ref-mask config where default GTM has known regions, but the queried
    # mask is unavailable in every segmentation -> supported_segs is empty.
    refmask_dir = tmp_path / 'fake_refmask'
    refmask_dir.mkdir()
    (refmask_dir / 'config.json').write_text(
        '{"gtm": {"cerebellum": {"refmask_indices": [47, 8]}}, "wm": {"semiovale": {"refmask_indices": [5001, 5002]}}}'
    )

    def _mock_files(pkg_name):
        if pkg_name == 'petprep.data.reference_mask':
            return refmask_dir
        return ir_files(pkg_name)

    monkeypatch.setattr(resources, 'files', _mock_files)

    with pytest.raises(SystemExit):
        parse_args(args=base_args + ['--ref-mask-name', 'not-a-region'])
    err = capsys.readouterr().err
    assert (
        "--ref-mask-name 'not-a-region' is not available for --seg gtm. "
        'Choose one of: cerebellum for --seg gtm.' in err
    )
    _reset_config()

    # Segmentation choices can include entries not present in refmask config.
    with pytest.raises(SystemExit):
        parse_args(args=base_args + ['--seg', 'brainstem', '--ref-mask-name', 'cerebellum'])
    err = capsys.readouterr().err
    assert '--seg brainstem does not define any predefined reference masks.' in err
    _reset_config()


def test_hmc_init_frame_parsing(tmp_path):
    """Ensure --hmc-init-frame accepts optional integers and defaults to auto."""
    datapath = tmp_path / 'data'
    outpath = tmp_path / 'out'
    datapath.mkdir()

    parser = _build_parser()
    base_args = [str(datapath), str(outpath), 'participant']

    opts = parser.parse_args(base_args)
    assert opts.hmc_init_frame == 'auto'

    opts = parser.parse_args(base_args + ['--hmc-init-frame'])
    assert opts.hmc_init_frame == 'auto'

    opts = parser.parse_args(base_args + ['--hmc-init-frame', '3', '--hmc-init-frame-fix'])
    assert opts.hmc_init_frame == 3
    assert opts.hmc_fix_frame is True


def test_hmc_off_flag(tmp_path):
    """Ensure disabling motion correction is parsed correctly."""
    datapath = tmp_path / 'data'
    outpath = tmp_path / 'out'
    datapath.mkdir()

    parser = _build_parser()
    base_args = [str(datapath), str(outpath), 'participant']

    opts = parser.parse_args(base_args)
    assert opts.hmc_off is False

    opts = parser.parse_args(base_args + ['--hmc-off'])
    assert opts.hmc_off is True
