# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
"""Helpers for FreeSurfer ``gtmseg`` outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _not_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return True
    return False


def _read_stats_table(stats_file: str | Path) -> pd.DataFrame:
    from pathlib import Path
    import pandas as pd
    """Parse a FreeSurfer ``*.stats`` file into a :class:`~pandas.DataFrame`."""
    stats_file = Path(stats_file)
    headers: list[str] | None = None
    rows: list[list[str]] = []
    first_row: list[str] | None = None
    with stats_file.open() as f:
        for line in f:
            if line.startswith('#'):
                stripped = line.strip('# \n')
                if 'ColHeaders' in stripped:
                    headers = stripped.split()[1:]
                elif headers is None:
                    tokens = stripped.split()
                    if {'Index', 'SegId', 'SegID', 'Name', 'StructName'} & set(tokens):
                        headers = tokens
                continue
            tokens = line.strip().split()
            if headers is None:
                if tokens and any(_not_number(t) for t in tokens):
                    headers = tokens
                else:
                    first_row = tokens
                continue
            if len(tokens) == len(headers):
                rows.append(tokens)
    if headers is None:
        raise ValueError(f'No table headers found in {stats_file}')
    if first_row is not None:
        if len(first_row) == len(headers):
            rows.insert(0, first_row)
        else:
            raise ValueError(f'Cannot parse stats table {stats_file}')
    return pd.DataFrame(rows, columns=headers)


def gtm_to_dsegtsv(subjects_dir: str, subject_id: str) -> str:
    """Generate a TSV table describing GTM segmentation labels."""
    from pathlib import Path

    import pandas as pd  # noqa: F401

    from petprep.utils.gtmseg import _read_stats_table

    gtm_stats = Path(subjects_dir) / subject_id / 'stats' / 'gtmseg.stats'
    df = _read_stats_table(gtm_stats)

    # Normalize column names for ease of access
    df.columns = [col.lower() for col in df.columns]

    if 'segid' in df.columns:
        segid = df.pop('segid')
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        df.insert(0, 'index', segid)
    elif 'index' not in df.columns:
        raise ValueError('No "segid" or "index" column found in stats table')

    name_col = 'name' if 'name' in df.columns else 'structname'

    df = df[['index', name_col]].rename(columns={name_col: 'name'})

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

    # Use the ``segid`` column when present and drop FreeSurfer's ``index``
    if 'segid' in df.columns:
        segid = df.pop('segid')
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        df.insert(0, 'index', segid)
    elif 'index' not in df.columns:
        raise ValueError('No "segid" or "index" column found in stats table')

    # Determine the column names for the region name and volume
    name_col = 'name' if 'name' in df.columns else 'structname'
    vol_col = 'volume_mm3' if 'volume_mm3' in df.columns else 'volume'

    df = df[['index', name_col, vol_col]].rename(
        columns={name_col: 'name', vol_col: 'volume-mm3'}
    )

    out_file = gtm_stats.with_name('gtmseg_morph.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)


def summary_to_stats(summary_file: str) -> str:
    """Convert a ``summary.stats`` file from ``mri_segstats`` to TSV."""
    from pathlib import Path

    from petprep.utils.gtmseg import _read_stats_table

    summary_file = Path(summary_file)
    df = _read_stats_table(summary_file)
    df.columns = [c.lower() for c in df.columns]

    if 'segid' in df.columns:
        segid = df.pop('segid')
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        df.insert(0, 'index', segid)
    elif 'index' not in df.columns:
        raise ValueError('No "segid" or "index" column found in stats table')

    name_col = 'name' if 'name' in df.columns else 'structname'
    vol_col = 'volume_mm3' if 'volume_mm3' in df.columns else 'volume'

    df = df[['index', name_col, vol_col]].rename(columns={name_col: 'name', vol_col: 'volume-mm3'})

    out_file = summary_file.with_suffix('.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)


def ctab_to_dsegtsv(ctab_file: str) -> str:
    """Convert a FreeSurfer ``ctab`` file to a TSV label table."""
    from pathlib import Path
    import pandas as pd

    ctab_file = Path(ctab_file)
    df = pd.read_csv(ctab_file, header=None, delim_whitespace=True, usecols=[0, 1], names=['index', 'name'])
    out_file = ctab_file.with_suffix('.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)
