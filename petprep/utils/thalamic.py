from __future__ import annotations

from .gtmseg import _read_stats_table


def summary_to_stats(summary_file: str | Path) -> str:
    """Convert a ``summary.stats`` file from ``mri_segstats`` to TSV."""
    from pathlib import Path

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


def ctab_to_dsegtsv(ctab_file: str | Path) -> str:
    """Convert a FreeSurfer ``ctab`` file to a TSV label table."""
    from pathlib import Path
    import pandas as pd

    ctab_file = Path(ctab_file)
    df = pd.read_csv(ctab_file, header=None, delim_whitespace=True, usecols=[0, 1], names=['index', 'name'])
    out_file = ctab_file.with_suffix('.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)
