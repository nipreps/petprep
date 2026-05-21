import subprocess
import sys

import petprep_docker.__main__ as ppd


def test_docker_get_parser():
    parser = ppd.get_parser()
    assert parser is not None


def test_merge_help_strips_ansi_codes(monkeypatch, tmp_path):
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mplconfig'))

    from petprep.cli.parser import _build_parser

    wrapper_help = ppd.get_parser().format_help()
    wrapper_help = wrapper_help.replace('[-h]', '[\x1b[1m-h\x1b[0m]', 1)
    wrapper_help = wrapper_help.replace('[--version]', '[\x1b[1m--version\x1b[0m]', 1)

    merged = ppd.merge_help(wrapper_help, _build_parser().format_help())

    assert merged.startswith('usage:')


def test_docker_main_help(monkeypatch, capsys):
    monkeypatch.setattr(ppd, 'check_docker', lambda: 1)
    monkeypatch.setattr(ppd, 'check_image', lambda img: True)
    monkeypatch.setattr(ppd, 'check_memory', lambda img: 16000)

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        # handle docker version query
        if '--format' in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=b'20.10')
        return subprocess.CompletedProcess(cmd, 0)

    def fake_check_output(cmd):
        captured['cmd'] = cmd
        return b'usage: petprep bids_dir output_dir {participant}\n\noptional arguments:'

    monkeypatch.setattr(ppd.subprocess, 'run', fake_run)
    monkeypatch.setattr(ppd.subprocess, 'check_output', fake_check_output)
    monkeypatch.setattr(ppd, 'merge_help', lambda a, b: 'merged')

    sys.argv = ['petprep-docker', '--help']
    ret = ppd.main()
    assert ret == 0
    assert captured['cmd'][:3] == ['docker', 'run', '--rm']
    assert '-h' in captured['cmd']
    assert '-i' in captured['cmd']
    assert ppd.__name__


def test_docker_main_version(monkeypatch):
    monkeypatch.setattr(ppd, 'check_docker', lambda: 1)
    monkeypatch.setattr(ppd, 'check_image', lambda img: True)
    monkeypatch.setattr(ppd, 'check_memory', lambda img: 16000)

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if '--format' in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=b'20.10')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ppd.subprocess, 'run', fake_run)

    sys.argv = ['petprep-docker', '--version']
    ret = ppd.main()
    assert ret == 0
    cmd = calls[-1]
    assert cmd[:3] == ['docker', 'run', '--rm']
    assert '--version' in cmd


def test_docker_command_options(monkeypatch, tmp_path):
    monkeypatch.setattr(ppd, 'check_docker', lambda: 1)
    monkeypatch.setattr(ppd, 'check_image', lambda img: True)
    monkeypatch.setattr(ppd, 'check_memory', lambda img: 16000)

    bids_dir = tmp_path / 'bids'
    out_dir = tmp_path / 'out'
    work_dir = tmp_path / 'work'
    bids_dir.mkdir()
    out_dir.mkdir()
    work_dir.mkdir()

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        if '--format' in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=b'20.10')
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ppd.subprocess, 'run', fake_run)

    sys.argv = [
        'petprep-docker',
        str(bids_dir),
        str(out_dir),
        'participant',
        '--work-dir',
        str(work_dir),
        '--output-spaces',
        'MNI152Lin',
    ]
    ret = ppd.main()
    assert ret == 0
    cmd = calls[-1]
    joined = ' '.join(cmd)
    assert f'{work_dir}:/scratch' in joined
    assert '--output-spaces' in cmd
    assert 'MNI152Lin' in cmd
