from pathlib import Path
from types import SimpleNamespace

from ... import config
from ..segmentation import (
    MRISclimbicSeg,
    SegmentBS,
    SegmentGTM,
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
