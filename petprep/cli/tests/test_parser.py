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
from contextlib import nullcontext

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
You are using fMRIPrep-{current}, and a newer version of fMRIPrep is available: {latest}.
Please check out our documentation about how and when to upgrade:
https://fmriprep.readthedocs.io/en/latest/faq.html#upgrading"""

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


@pytest.mark.parametrize('st_ref', (None, '0', '1', '0.5', 'start', 'middle'))  # noqa: PT007
def test_slice_time_ref(tmp_path, st_ref):
    bids_path = tmp_path / 'data'
    out_path = tmp_path / 'out'
    args = [str(bids_path), str(out_path), 'participant']
    if st_ref:
        args.extend(['--slice-time-ref', st_ref])
    bids_path.mkdir()

    parser = _build_parser()

    parser.parse_args(args)
    _reset_config()


@pytest.mark.parametrize(
    ('args', 'expectation'),
    [
        ([], False),
        (['--use-syn-sdc'], 'error'),
        (['--use-syn-sdc', 'error'], 'error'),
        (['--use-syn-sdc', 'warn'], 'warn'),
        (['--use-syn-sdc', 'other'], (SystemExit, ArgumentError)),
    ],
)
def test_use_syn_sdc(tmp_path, args, expectation):
    bids_path = tmp_path / 'data'
    out_path = tmp_path / 'out'
    args = [str(bids_path), str(out_path), 'participant'] + args
    bids_path.mkdir()

    parser = _build_parser()

    cm = nullcontext()
    if isinstance(expectation, tuple):
        cm = pytest.raises(expectation)

    with cm:
        opts = parser.parse_args(args)

    if not isinstance(expectation, tuple):
        assert opts.use_syn_sdc == expectation

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
