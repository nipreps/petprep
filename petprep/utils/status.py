"""Utilities for summarizing participant processing outcomes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

STATUS_COLUMNS = [
    'participant',
    'status',
    'has_pet',
    'has_t1w',
    'has_native_t1w',
    'used_derivative_t1w',
    'n_sessions',
    'n_pet_runs',
    'missing_inputs',
    'crash_files',
    'notes',
]


def collect_participant_status(
    petprep_dir: Path | str,
    run_uuid: str,
    *,
    participants: Iterable[str] | None = None,
    metadata: Mapping[str, MutableMapping[str, object]] | None = None,
    failed_reports: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Gather per-participant processing outcomes and return report rows."""

    petprep_dir = Path(petprep_dir)
    metadata = metadata or {}
    failed = {str(label).removeprefix('sub-') for label in (failed_reports or [])}

    normalized_participants: list[str] = []
    seen: set[str] = set()
    if participants:
        for label in participants:
            clean = str(label).removeprefix('sub-')
            if clean not in seen:
                normalized_participants.append(clean)
                seen.add(clean)
    else:
        normalized_participants.extend(sorted(metadata.keys()))
        seen.update(metadata.keys())

    for label in sorted(metadata.keys()):
        if label not in seen:
            normalized_participants.append(label)
            seen.add(label)

    rows: list[dict[str, str]] = []
    for label in normalized_participants:
        record = metadata.get(label, {})
        has_pet = _coerce_bool(record.get('has_pet'))
        native_t1w = _coerce_bool(record.get('has_native_t1w'))
        derivative_t1w = _coerce_bool(record.get('has_derivative_t1w'))
        has_any_t1w = record.get('has_any_t1w')
        if has_any_t1w is None:
            has_any_t1w = (native_t1w is True) or (derivative_t1w is True)
        else:
            has_any_t1w = bool(has_any_t1w)

        missing_inputs: list[str] = []
        if has_any_t1w is False:
            missing_inputs.append('T1w')
        if has_pet is False and not _coerce_bool(record.get('anat_only')):
            missing_inputs.append('PET')

        crash_dir = petprep_dir / f'sub-{label}' / 'log' / run_uuid
        crash_files = sorted(p.name for p in crash_dir.glob('crash*.*')) if crash_dir.exists() else []

        status = 'completed'
        if crash_files or label in failed:
            status = 'failed'
        elif missing_inputs:
            status = 'skipped'

        notes = record.get('notes') or []
        if isinstance(notes, str):
            notes_list = [notes]
        else:
            notes_list = list(notes)

        row = {
            'participant': f'sub-{label}',
            'status': status,
            'has_pet': _format_bool(has_pet),
            'has_t1w': _format_bool(has_any_t1w),
            'has_native_t1w': _format_bool(native_t1w),
            'used_derivative_t1w': _format_bool(derivative_t1w),
            'n_sessions': _format_int(record.get('n_sessions')),
            'n_pet_runs': _format_int(record.get('n_pet_runs')),
            'missing_inputs': ','.join(missing_inputs),
            'crash_files': ','.join(crash_files),
            'notes': '; '.join(notes_list),
        }
        rows.append(row)

    return rows


def write_participant_log(
    log_dir: Path | str,
    run_uuid: str,
    rows: Sequence[Mapping[str, str]],
) -> Path:
    """Write the participant status log to disk and return the path."""

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    output = log_path / f'run-{run_uuid}_participant-status.tsv'

    with output.open('w', newline='') as fobj:
        writer = csv.DictWriter(fobj, fieldnames=STATUS_COLUMNS, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, '') for column in STATUS_COLUMNS})

    return output


def _format_bool(value: bool | None) -> str:
    if value is None:
        return ''
    return 'True' if value else 'False'


def _format_int(value: object) -> str:
    if value in (None, ''):
        return ''
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ''


def _coerce_bool(value: object) -> bool | None:
    if value in (None, ''):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {'true', 'yes', '1'}:
            return True
        if value.lower() in {'false', 'no', '0'}:
            return False
    try:
        return bool(value)
    except Exception:  # noqa: BLE001
        return None
