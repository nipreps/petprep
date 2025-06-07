# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Helpers for FreeSurfer ``gtmseg`` outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_stats_table(stats_file: str | Path) -> pd.DataFrame:
    from pathlib import Path
    import pandas as pd
    """Parse a FreeSurfer ``*.stats`` file into a :class:`~pandas.DataFrame`."""
    stats_file = Path(stats_file)
    headers: list[str] | None = None
    rows: list[list[str]] = []
    with stats_file.open() as f:
        for line in f:
            if line.startswith('#'):
                if 'ColHeaders' in line:
                    headers = line.strip('# \n').split()[1:]
                continue
            if headers:
                parts = line.strip().split()
                if len(parts) == len(headers):
                    rows.append(parts)
    if headers is None:
        raise ValueError(f'No table headers found in {stats_file}')
    return pd.DataFrame(rows, columns=headers)


def gtm_to_dsegtsv(subjects_dir: str, subject_id: str) -> str:
    """Generate a TSV table summarizing GTM segmentation volumes."""
    from pathlib import Path

    import pandas as pd  # noqa: F401

    from petprep.utils.gtmseg import _read_stats_table

    gtm_stats = Path(subjects_dir) / subject_id / 'stats' / 'gtmseg.stats'
    df = _read_stats_table(gtm_stats)
    out_file = gtm_stats.with_name('gtmseg_dseg.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)


def gtm_stats_to_stats(subjects_dir: str, subject_id: str) -> str:
    """Generate a TSV table of morphological statistics from ``gtmseg.stats``."""
    from pathlib import Path

    import pandas as pd  # noqa: F401

    from petprep.utils.gtmseg import _read_stats_table

    gtm_stats = Path(subjects_dir) / subject_id / 'stats' / 'gtmseg.stats'
    df = _read_stats_table(gtm_stats)
    out_file = gtm_stats.with_name('gtmseg_morph.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)