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
from pathlib import Path

import pandas as pd
from nireports.assembler.report import Report

from .. import config, data


def run_reports(
    output_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    out_filename='report.html',
    reportlets_dir=None,
    errorname='report.err',
    **entities,
):
    """
    Run the reports.
    """
    robj = Report(
        output_dir,
        run_uuid,
        bootstrap_file=bootstrap_file,
        out_filename=out_filename,
        reportlets_dir=reportlets_dir,
        plugins=None,
        plugin_meta=None,
        metadata=None,
        **entities,
    )

    # Count nbr of subject for which report generation failed
    try:
        robj.generate_report()
    except:  # noqa: E722
        import sys
        import traceback

        # Store the list of subjects for which report generation failed
        traceback.print_exception(*sys.exc_info(), file=str(Path(output_dir) / 'logs' / errorname))
        return subject_label

    return None


def generate_reports(
    subject_list, output_dir, run_uuid, session_list=None, bootstrap_file=None, work_dir=None
):
    """Generate reports for a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / 'reportlets'

    if isinstance(subject_list, str):
        subject_list = [subject_list]
    if isinstance(session_list, str):
        session_list = [session_list]

    errors = []
    for subject_label in subject_list:
        subject_label = subject_label.removeprefix('sub-')

        subject_reportlets_dir = reportlets_dir
        if reportlets_dir is not None:
            subject_reportlets_dir = next(
                (
                    candidate
                    for candidate in (
                        reportlets_dir / 'petprep' / f'sub-{subject_label}',
                        reportlets_dir / f'sub-{subject_label}',
                        reportlets_dir,
                    )
                    if candidate.exists()
                ),
                reportlets_dir,
            )
        # The number of sessions is intentionally not based on session_list but
        # on the total number of sessions, because I want the final derivatives
        # folder to be the same whether sessions were run one at a time or all-together.
        n_ses = len(config.execution.layout.get_sessions(subject=subject_label))

        if bootstrap_file is not None:
            # If a config file is precised, we do not override it
            html_report = 'report.html'
        elif n_ses <= config.execution.aggr_ses_reports:
            # If there are only a few session for this subject,
            # we aggregate them in a single visual report.
            bootstrap_file = data.load('reports-spec.yml')
            html_report = 'report.html'
        else:
            # Beyond a threshold, we separate the anatomical report from the PET.
            bootstrap_file = data.load('reports-spec-anat.yml')
            html_report = f'sub-{subject_label}_anat.html'

        report_error = run_reports(
            output_dir,
            subject_label,
            run_uuid,
            bootstrap_file=bootstrap_file,
            out_filename=html_report,
            reportlets_dir=subject_reportlets_dir,
            errorname=f'report-{run_uuid}-{subject_label}.err',
            subject=subject_label,
        )
        # If the report generation failed, append the subject label for which it failed
        if report_error is not None:
            errors.append(report_error)

        if n_ses > config.execution.aggr_ses_reports:
            # Beyond a certain number of sessions per subject,
            # we separate the PET reports per session
            if session_list is None:
                all_filters = config.execution.bids_filters or {}
                filters = all_filters.get('pet', {})
                session_list = config.execution.layout.get_sessions(
                    subject=subject_label, **filters
                )

            session_list = [ses.removeprefix('ses-') for ses in session_list]

            for session_label in session_list:
                bootstrap_file = data.load('reports-spec-pet.yml')
                html_report = f'sub-{subject_label}_ses-{session_label}_pet.html'

                report_error = run_reports(
                    output_dir,
                    subject_label,
                    run_uuid,
                    bootstrap_file=bootstrap_file,
                    out_filename=html_report,
                    reportlets_dir=reportlets_dir,
                    errorname=f'report-{run_uuid}-{subject_label}-pet.err',
                    subject=subject_label,
                    session=session_label,
                )
                # If the report generation failed, append the subject label for which it failed
                if report_error is not None:
                    errors.append(report_error)

                bootstrap_file = data.load('reports-spec-pet.yml')
                html_report = f'sub-{subject_label}_ses-{session_label}_pet.html'

                report_error = run_reports(
                    output_dir,
                    subject_label,
                    run_uuid,
                    bootstrap_file=bootstrap_file,
                    out_filename=html_report,
                    reportlets_dir=reportlets_dir,
                    errorname=f'report-{run_uuid}-{subject_label}-pet.err',
                    subject=subject_label,
                    session=session_label,
                )
                if report_error is not None:
                    errors.append(report_error)

    return errors


def generate_group_morph_report(output_dir, participant_label=None):
    """Generate group-level summaries from participant morphometry and timeseries files."""
    output_dir = Path(output_dir)
    group_dir = output_dir / 'group'
    group_dir.mkdir(parents=True, exist_ok=True)

    if participant_label:
        participants = [label.removeprefix('sub-') for label in participant_label]
    else:
        participants = sorted(path.name.removeprefix('sub-') for path in output_dir.glob('sub-*'))

    morph_dfs = []
    for subject_id in participants:
        sub_dir = output_dir / f'sub-{subject_id}'
        if not sub_dir.exists():
            continue

        for morph_file in sub_dir.rglob('*_morph.tsv'):
            if 'anat' not in morph_file.parts:
                continue
            sub_df = pd.read_csv(morph_file, sep='\t')
            sub_df['participant_id'] = f'sub-{subject_id}'
            sub_df['source_file'] = str(morph_file.relative_to(output_dir))
            morph_dfs.append(sub_df)

    if not morph_dfs:
        raise RuntimeError(f'No participant files (*_morph.tsv) were found under {output_dir}.')

    group_df = pd.concat(morph_dfs, ignore_index=True)
    numeric_cols = [
        col
        for col in group_df.select_dtypes(include='number').columns
        if col not in {'index', 'id'}
    ]
    if not numeric_cols:
        raise RuntimeError('No numeric columns were found in participant morphometry tables.')

    groupby_cols = [
        col
        for col in group_df.columns
        if col not in set(numeric_cols) | {'participant_id', 'source_file'}
    ]

    summary_df = (
        group_df.groupby(groupby_cols, dropna=False)[numeric_cols]
        .agg(['count', 'mean', 'std', 'min', 'max'])
        .reset_index()
    )
    summary_df.columns = [
        '_'.join(filter(None, map(str, col))).rstrip('_') for col in summary_df.columns.to_flat_index()
    ]

    summary_tsv = group_dir / 'desc-morph_group.tsv'
    confounds_tsv = group_dir / 'desc-timeseries_group.tsv'
    summary_html = group_dir / 'report.html'

    summary_df.to_csv(summary_tsv, sep='\t', index=False)

    timeseries_dfs = []
    for subject_id in participants:
        sub_dir = output_dir / f'sub-{subject_id}'
        if not sub_dir.exists():
            continue

        for timeseries_file in sub_dir.rglob('*_timeseries.tsv'):
            sub_df = pd.read_csv(timeseries_file, sep='\t')
            numeric_cols = list(sub_df.select_dtypes(include='number').columns)
            if not numeric_cols:
                continue
            timeseries_dfs.append(sub_df[numeric_cols])

    confounds_summary_df = pd.DataFrame()
    if timeseries_dfs:
        confounds_df = pd.concat(timeseries_dfs, ignore_index=True)
        confounds_summary_df = (
            confounds_df.agg(['count', 'mean', 'std', 'min', 'max']).transpose().reset_index()
        )
        confounds_summary_df = confounds_summary_df.rename(columns={'index': 'name'})
        confounds_summary_df.to_csv(confounds_tsv, sep='\t', index=False)

    summary_html.write_text(
        '\n'.join(
            [
                '<html><head><meta charset="utf-8"><title>PETPrep Group Morphometry Report</title></head><body>',
                '<h1>PETPrep group morphometry summary</h1>',
                '<p>Summary statistics across participant-level <code>*_morph.tsv</code> outputs.</p>',
                summary_df.to_html(index=False, border=0),
                '<h2>PETPrep group timeseries summary</h2>',
                (
                    '<p>Summary statistics across numeric columns from participant-level '
                    '<code>*_timeseries.tsv</code> outputs.</p>'
                ),
                (
                    confounds_summary_df.to_html(index=False, border=0)
                    if not confounds_summary_df.empty
                    else '<p>No numeric <code>*_timeseries.tsv</code> files were found.</p>'
                ),
                '</body></html>',
            ]
        )
    )

    return summary_tsv, summary_html
