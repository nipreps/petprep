from pathlib import Path
from types import SimpleNamespace

from ... import config
from ..segmentation import (
    MRISclimbicSeg,
    SegmentBS,
    SegmentCC,
    SegmentGTM,
    SegmentHA_T1,
    SegmentThalamicNuclei,
    SegmentWM,
    _ensure_mcr2019b_installed,
    _set_freesurfer_seed,
)


def test_segmentgtm_skip(tmp_path):
    subj_dir = tmp_path / 'sub-01'
    (subj_dir / 'mri').mkdir(parents=True)
    (subj_dir / 'stats').mkdir()
    (subj_dir / 'mri' / 'gtmseg.mgz').write_text('')
    (subj_dir / 'stats' / 'gtmseg.stats').write_text('')

    seg = SegmentGTM(subjects_dir=str(tmp_path), subject_id='sub-01')
    res = seg.run()

    assert res.runtime.returncode == 0
    assert Path(res.outputs.out_file) == subj_dir / 'mri' / 'gtmseg.mgz'
    assert res.runtime.environ['FREESURFER_RANDOM_SEED'] == str(config.seeds.freesurfer)


def test_mrisclimbicseg_seed(tmp_path):
    subjects_dir = tmp_path / 'subjects'
    subject_dir = subjects_dir / 'sub-01'
    subject_dir.mkdir(parents=True)

    out_file = subject_dir / 'sub-01_sclimbic.nii.gz'
    out_stats = subject_dir / 'sub-01_sclimbic.stats'
    out_file.write_text('')
    out_stats.write_text('')

    seg = MRISclimbicSeg(out_file=str(out_file), sd=str(subjects_dir), subjects=['sub-01'])
    res = seg.run()

    assert res.runtime.returncode == 0
    assert res.runtime.environ['FREESURFER_RANDOM_SEED'] == str(config.seeds.freesurfer)


def _fake_bs_run(self, cmd):
    subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri'
    subj_dir.mkdir(parents=True, exist_ok=True)
    (subj_dir / 'brainstemSsLabels.v13.mgz').write_text('')
    (subj_dir / 'brainstemSsLabels.v13.FSvoxelSpace.mgz').write_text('')
    (subj_dir / 'brainstemSsVolumes.v13.txt').write_text('')
    return 'bs out', 'bs err'


def _fake_wm_run(self, cmd):
    subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri'
    subj_dir.mkdir(parents=True, exist_ok=True)
    (subj_dir / 'wmparc.mgz').write_text('')
    return 'wm out', 'wm err'


def _fake_cc_run(self, cmd):
    subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri'
    subj_dir.mkdir(parents=True, exist_ok=True)
    (subj_dir / self.inputs.out_file).write_text('')
    self._cmd = cmd
    return 'cc out', 'cc err'


def test_segmentbs_stdout_stderr(monkeypatch, tmp_path):
    seg = SegmentBS(subjects_dir=str(tmp_path), subject_id='sub-01')
    monkeypatch.setattr(SegmentBS, '_run_command', _fake_bs_run)
    monkeypatch.setattr(
        'petprep.interfaces.segmentation._ensure_mcr2019b_installed',
        lambda runtime: runtime,
    )
    res = seg.run()
    assert res.outputs.stdout == 'bs out'
    assert res.outputs.stderr == 'bs err'


def test_segmentwm_stdout_stderr(monkeypatch, tmp_path):
    seg = SegmentWM(subjects_dir=str(tmp_path), subject_id='sub-01')
    monkeypatch.setattr(SegmentWM, '_run_command', _fake_wm_run)
    res = seg.run()
    assert res.outputs.stdout == 'wm out'
    assert res.outputs.stderr == 'wm err'


def test_segmentcc_stdout_stderr_and_command(monkeypatch, tmp_path):
    seg = SegmentCC(
        subjects_dir=str(tmp_path),
        subject_id='sub-01',
        force=True,
        subdivisions=5,
        thickness=2,
    )
    monkeypatch.setattr(SegmentCC, '_run_command', _fake_cc_run)
    res = seg.run()

    assert res.outputs.stdout == 'cc out'
    assert res.outputs.stderr == 'cc err'
    assert Path(res.outputs.out_file) == tmp_path / 'sub-01' / 'mri' / 'aseg.auto_CCseg.mgz'
    assert seg._cmd == [
        'mri_cc',
        '-aseg',
        'aseg.mgz',
        '-o',
        'aseg.auto_CCseg.mgz',
        '-sdir',
        str(tmp_path),
        '-force',
        '-d',
        '5',
        '-t',
        '2',
        'sub-01',
    ]


def test_set_freesurfer_seed_runtime():
    runtime = SimpleNamespace(environ={})

    runtime = _set_freesurfer_seed(runtime)

    assert runtime.environ['FREESURFER_RANDOM_SEED'] == str(config.seeds.freesurfer)


def test_segmentbs_skips_mcr_when_outputs_exist(monkeypatch, tmp_path):
    subj_dir = tmp_path / 'sub-01' / 'mri'
    subj_dir.mkdir(parents=True)
    (subj_dir / 'brainstemSsLabels.v13.mgz').write_text('')
    (subj_dir / 'brainstemSsLabels.v13.FSvoxelSpace.mgz').write_text('')
    (subj_dir / 'brainstemSsVolumes.v13.txt').write_text('')

    seg = SegmentBS(subjects_dir=str(tmp_path), subject_id='sub-01')

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError('MCR install should not run when outputs already exist.')

    monkeypatch.setattr(
        'petprep.interfaces.segmentation._ensure_mcr2019b_installed', _raise_if_called
    )
    res = seg.run()
    assert res.runtime.returncode == 0


def test_segment_thalamic_installs_mcr_before_running(monkeypatch, tmp_path):
    seg = SegmentThalamicNuclei(subjects_dir=str(tmp_path), subject_id='sub-01')
    calls = {'mcr': 0, 'run': 0}

    def _fake_mcr(runtime):
        calls['mcr'] += 1
        return runtime

    def _fake_run(self, _cmd):
        calls['run'] += 1
        subj_dir = Path(self.inputs.subjects_dir) / self.inputs.subject_id / 'mri'
        subj_dir.mkdir(parents=True, exist_ok=True)
        (subj_dir / 'ThalamicNuclei.v13.T1.FSvoxelSpace.mgz').write_text('')
        (subj_dir / 'ThalamicNuclei.v13.T1.volumes.txt').write_text('')

    monkeypatch.setattr('petprep.interfaces.segmentation._ensure_mcr2019b_installed', _fake_mcr)
    monkeypatch.setattr(SegmentThalamicNuclei, '_run_command', _fake_run)
    seg.run()

    assert calls['mcr'] == 1
    assert calls['run'] == 1


def test_mcr_lookup_uses_freesurfer_home_not_mcrroot(monkeypatch, tmp_path):
    fs_home = tmp_path / 'freesurfer'
    (fs_home / 'MCRv97').mkdir(parents=True)

    runtime = SimpleNamespace(
        environ={'FREESURFER_HOME': str(fs_home), 'MCRROOT': '/tmp/does-not-exist'}
    )

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError(
            'MCR installer should not run when MCRv97 exists under FREESURFER_HOME.'
        )

    monkeypatch.setattr('subprocess.run', _raise_if_called)

    result = _ensure_mcr2019b_installed(runtime)

    assert result is runtime


def test_mcr_install_raises_when_mcr_missing(tmp_path):
    fs_home = tmp_path / 'freesurfer'
    fs_home.mkdir()

    runtime = SimpleNamespace(environ={'FREESURFER_HOME': str(fs_home)})

    try:
        _ensure_mcr2019b_installed(runtime)
    except RuntimeError as err:
        assert 'MCRv97' in str(err)
    else:
        raise AssertionError('Expected RuntimeError when MCRv97 is missing.')


def test_segment_thalamic_skips_when_outputs_exist(monkeypatch, tmp_path):
    subj_dir = tmp_path / 'sub-01' / 'mri'
    subj_dir.mkdir(parents=True)
    (subj_dir / 'ThalamicNuclei.v13.T1.FSvoxelSpace.mgz').write_text('')
    (subj_dir / 'ThalamicNuclei.v13.T1.volumes.txt').write_text('')

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError('Segmentation command should not run when outputs already exist.')

    monkeypatch.setattr(SegmentThalamicNuclei, '_run_command', _raise_if_called)

    seg = SegmentThalamicNuclei(subjects_dir=str(tmp_path), subject_id='sub-01')
    res = seg.run()

    assert res.runtime.returncode == 0
    assert Path(res.outputs.out_file) == subj_dir / 'ThalamicNuclei.v13.T1.FSvoxelSpace.mgz'
    assert Path(res.outputs.volumes_file) == subj_dir / 'ThalamicNuclei.v13.T1.volumes.txt'


def test_segmentcc_skips_when_output_exists(monkeypatch, tmp_path):
    subj_dir = tmp_path / 'sub-01' / 'mri'
    subj_dir.mkdir(parents=True)
    (subj_dir / 'aseg.auto_CCseg.mgz').write_text('')

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError('Corpus callosum command should not run when output exists.')

    monkeypatch.setattr(SegmentCC, '_run_command', _raise_if_called)

    seg = SegmentCC(subjects_dir=str(tmp_path), subject_id='sub-01')
    res = seg.run()

    assert res.runtime.returncode == 0
    assert Path(res.outputs.out_file) == subj_dir / 'aseg.auto_CCseg.mgz'


def test_segmentha_t1_skip_and_filename(tmp_path):
    subj_dir = tmp_path / 'sub-01' / 'mri'
    subj_dir.mkdir(parents=True)
    expected_files = [
        'lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz',
        'rh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz',
        'lh.hippoSfVolumes-T1.v22.txt',
        'lh.amygNucVolumes-T1.v22.txt',
        'rh.hippoSfVolumes-T1.v22.txt',
        'rh.amygNucVolumes-T1.v22.txt',
    ]
    for fname in expected_files:
        (subj_dir / fname).write_text('')

    seg = SegmentHA_T1(subjects_dir=str(tmp_path), subject_id='sub-01')
    res = seg.run()

    assert res.runtime.returncode == 0
    assert Path(res.outputs.lh_hippoAmygLabels) == subj_dir / expected_files[0]
    assert Path(res.outputs.rh_amygNucVolumes) == subj_dir / expected_files[-1]
    assert seg._gen_filename('subjects_dir') == str((Path.cwd() / 'sub-01').resolve())
    assert seg._gen_filename('not-a-real-field') is None
