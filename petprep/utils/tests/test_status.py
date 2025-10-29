from petprep.utils.status import collect_participant_status, write_participant_log


def test_collect_and_write_participant_status(tmp_path):
    run_uuid = 'test-uuid'

    metadata = {
        '01': {
            'anat_only': False,
            'has_pet': True,
            'has_native_t1w': True,
            'has_derivative_t1w': False,
            'has_any_t1w': True,
            'n_sessions': 2,
            'n_pet_runs': 3,
            'notes': [],
        },
        '02': {
            'anat_only': False,
            'has_pet': True,
            'has_native_t1w': False,
            'has_derivative_t1w': False,
            'has_any_t1w': False,
            'n_sessions': 1,
            'n_pet_runs': 1,
            'notes': ['Missing anatomical images'],
        },
        '03': {
            'anat_only': False,
            'has_pet': True,
            'has_native_t1w': True,
            'has_derivative_t1w': False,
            'has_any_t1w': True,
            'n_sessions': 1,
            'n_pet_runs': 1,
            'notes': [],
        },
    }

    crash_dir = tmp_path / 'sub-03' / 'log' / run_uuid
    crash_dir.mkdir(parents=True)
    (crash_dir / 'crash-test.txt').write_text('boom')

    rows = collect_participant_status(
        tmp_path,
        run_uuid,
        participants=['sub-01', 'sub-02', 'sub-03'],
        metadata=metadata,
        failed_reports=['03'],
    )

    assert [row['participant'] for row in rows] == ['sub-01', 'sub-02', 'sub-03']
    assert rows[0]['status'] == 'completed'
    assert rows[1]['status'] == 'skipped'
    assert rows[1]['missing_inputs'] == 'T1w'
    assert rows[2]['status'] == 'failed'
    assert 'crash-test.txt' in rows[2]['crash_files']

    log_path = write_participant_log(tmp_path / 'logs', run_uuid, rows)

    assert log_path.exists()
    content = log_path.read_text().splitlines()
    assert content[0].startswith('participant\tstatus')
    assert any('sub-02' in line and 'skipped' in line for line in content[1:])
    assert any('sub-03' in line and 'failed' in line for line in content[1:])
