import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from bids.layout import BIDSLayout

import petprep.reports.core as core
from petprep.reports.core import generate_reports

from ... import config, data

data_dir = data.load('tests')
pet_source = data_dir / 'work' / 'reportlets' / 'petprep' / 'sub-01' / 'pet'


# Test with and without sessions' aggregation
@pytest.mark.parametrize(
    ('aggr_ses_reports', 'expected_files'),
    [
        (
            3,
            [
                'sub-01_anat.html',
                'sub-01_ses-001_pet.html',
                'sub-01_ses-003_pet.html',
                'sub-01_ses-004_pet.html',
                'sub-01_ses-005_pet.html',
            ],
        ),
        (4, ['sub-01.html']),
    ],
)
# Test with and without crash file
@pytest.mark.parametrize('error', [True, False])
# Test with and without boilerplate
@pytest.mark.parametrize('boilerplate', [True, False])
# Test ses- prefix stripping
@pytest.mark.parametrize(
    'session_list', [['001', '003', '004', '005'], ['ses-001', 'ses-003', 'ses-004', 'ses-005']]
)
# Test sub- prefix stripping
@pytest.mark.parametrize('subject_label', ['01', 'sub-01'])
@pytest.mark.skipif(
    not Path.exists(data_dir / 'work'),
    reason='Package installed - large test data directory excluded from wheel',
)
def test_ReportSeparation(
    tmp_path,
    monkeypatch,
    aggr_ses_reports,
    expected_files,
    error,
    boilerplate,
    session_list,
    subject_label,
):
    fake_uuid = 'fake_uuid'

    sub_dir = tmp_path / 'sub-01'
    shutil.copytree(data_dir / 'work/reportlets/petprep/sub-01', sub_dir)

    # Test report generation with and without crash file
    if error:
        crash_file = next(data_dir.glob('crash_files/crash*.txt'))
        run_log_dir = sub_dir / 'log' / fake_uuid
        run_log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(crash_file, run_log_dir / crash_file.name)

    # Test report generation with and without boilerplate
    if boilerplate:
        log_dir = tmp_path / 'logs'
        log_dir.mkdir()
        shutil.copy2(data_dir / 'logs/CITATION.html', log_dir / 'CITATION.html')

    monkeypatch.setattr(config.execution, 'aggr_ses_reports', aggr_ses_reports)

    def mock_session_list(*args, **kwargs):
        return session_list

    config.execution.layout = BIDSLayout(data_dir / 'ds000005')
    monkeypatch.setattr(config.execution.layout, 'get_sessions', mock_session_list)
    monkeypatch.setattr(
        config.execution, 'bids_filters', {'pet': {'session': ['001', '003', '004', '005']}}
    )

    # Generate report
    failed_reports = generate_reports([subject_label], tmp_path, fake_uuid)

    # Verify that report generation was successful
    assert not failed_reports

    # Check that all expected files were generated
    for expected_file in expected_files:
        file_path = tmp_path / expected_file
        assert file_path.is_file(), f'Expected file {expected_file} is missing'

    # Check if there are no unexpected HTML files
    unexpected_files = {file.name for file in tmp_path.glob('*.html')} - set(expected_files)
    assert not unexpected_files, f'Unexpected HTML files found: {unexpected_files}'

    if not (boilerplate or error):
        return

    html_content = Path.read_text(tmp_path / expected_files[0])
    if boilerplate:
        assert 'The boilerplate text was automatically generated' in html_content, (
            f'The file {expected_files[0]} did not contain the reported error.'
        )

    if error:
        assert 'One or more execution steps failed' in html_content, (
            f'The file {expected_files[0]} did not contain the reported error.'
        )


@pytest.mark.skipif(
    not pet_source.exists(),
    reason='Package installed - large test data directory excluded from wheel',
)
def test_pet_report(tmp_path, monkeypatch):
    fake_uuid = 'fake_uuid'

    sub_dir = tmp_path / 'sub-01' / 'figures'
    sub_dir.mkdir(parents=True)

    if not pet_source.exists():
        pytest.skip('Package installed - large test data directory excluded from wheel')

    for fl in [
        'sub-01_desc-about_T1w.html',
        'sub-01_desc-summary_pet.html',
        'sub-01_desc-validation_pet.html',
        'sub-01_desc-carpetplot_pet.svg',
        'sub-01_desc-confoundcorr_pet.svg',
        'sub-01_desc-coreg_pet.svg',
        'sub-01_desc-atlasrois_seg-HOCPA_pet.svg',
    ]:
        shutil.copy2(pet_source / fl, sub_dir / fl)

    monkeypatch.setattr(config.execution, 'aggr_ses_reports', 4)
    monkeypatch.setattr(config.execution, 'layout', BIDSLayout(data_dir / 'ds000005'))
    monkeypatch.setattr(config.execution, 'bids_filters', {'pet': {'session': ['baseline']}})

    failed_reports = generate_reports(['01'], tmp_path, fake_uuid)

    assert not failed_reports
    html_file = tmp_path / 'sub-01.html'
    assert html_file.is_file()
    html_content = html_file.read_text()
    assert '<div id="PET"' in html_content


def test_reportlets_dir_scoped_to_subject(tmp_path, monkeypatch):
    work_dir = tmp_path / 'work'
    target = work_dir / 'reportlets' / 'petprep' / 'sub-02'
    target.mkdir(parents=True)
    # Add similarly-named subjects to verify exact matches
    (work_dir / 'reportlets' / 'petprep' / 'sub-2').mkdir(parents=True)
    (work_dir / 'reportlets' / 'petprep' / 'sub-002').mkdir(parents=True)
    (work_dir / 'reportlets' / 'petprep' / 'sub-020').mkdir(parents=True)

    recorded_reportlets = []

    def _record_reportlets_dir(*args, reportlets_dir=None, **kwargs):
        recorded_reportlets.append(reportlets_dir)
        return None

    monkeypatch.setattr(core, 'run_reports', _record_reportlets_dir)
    monkeypatch.setattr(config.execution, 'aggr_ses_reports', 4)

    class _Layout:
        def get_sessions(self, subject=None):
            return []

    monkeypatch.setattr(config.execution, 'layout', _Layout())

    generate_reports(['02'], tmp_path, 'fake', work_dir=work_dir)

    assert recorded_reportlets == [target]


def test_generate_reports_uses_indexed_session_entities(tmp_path, monkeypatch):
    recorded_sessions = []

    def _record_report(*args, session=None, **kwargs):
        recorded_sessions.append(session)
        return None

    class _Layout:
        def get(self, return_type='object', subject=None, session=None, **kwargs):
            assert return_type == 'object'
            labels = ['001', '003']
            if session is not None:
                allowed = session if isinstance(session, list) else [session]
                labels = [label for label in labels if label in allowed]
            return [SimpleNamespace(entities={'session': label}) for label in labels]

    monkeypatch.setattr(core, 'run_reports', _record_report)
    monkeypatch.setattr(config.execution, 'aggr_ses_reports', 1)
    monkeypatch.setattr(config.execution, 'layout', _Layout())
    monkeypatch.setattr(config.execution, 'bids_filters', {'pet': {'session': ['001', '003']}})

    failed_reports = generate_reports(['01'], tmp_path, 'fake')

    assert not failed_reports
    assert recorded_sessions == [None, '001', '001', '003', '003']
