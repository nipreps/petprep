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

    # Normalize column names to lowercase for easier matching
    df.columns = [col.lower() for col in df.columns]

    # Map FreeSurfer "segid" to BIDS-compliant "index"
    if 'segid' in df.columns:
        df = df.rename(columns={'segid': 'index'})

    # Determine the column names for the region name and volume
    name_col = 'name' if 'name' in df.columns else 'structname'
    vol_col = 'volume_mm3' if 'volume_mm3' in df.columns else 'volume'

    df = df[['index', name_col, vol_col]].rename(
        columns={name_col: 'name', vol_col: 'volume-mm3'}
    )

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

    # Normalize column names to lowercase for easier matching
    df.columns = [col.lower() for col in df.columns]

    # Map FreeSurfer "segid" to BIDS-compliant "index"
    if 'segid' in df.columns:
        df = df.rename(columns={'segid': 'index'})

    # Determine the column names for the region name and volume
    name_col = 'name' if 'name' in df.columns else 'structname'
    vol_col = 'volume_mm3' if 'volume_mm3' in df.columns else 'volume'

    df = df[['index', name_col, vol_col]].rename(
        columns={name_col: 'name', vol_col: 'volume-mm3'}
    )

    out_file = gtm_stats.with_name('gtmseg_morph.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)