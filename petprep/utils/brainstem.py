from __future__ import annotations

from pathlib import Path
from typing import Iterable


def brainstem_to_dsegtsv(subjects_dir: str, subject_id: str) -> str:
    """Generate a TSV table describing brainstem segmentation labels."""
    from pathlib import Path

    from petprep.utils.gtmseg import _read_stats_table
    stats_file = Path(subjects_dir) / subject_id / 'stats' / 'brainstem.v13.stats'
    df = _read_stats_table(stats_file)
    df.columns = [c.lower() for c in df.columns]

    if 'segid' in df.columns:
        segid = df.pop('segid')
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        df.insert(0, 'index', segid)
    elif 'index' not in df.columns:
        raise ValueError('No "segid" or "index" column found in stats table')

    name_col = 'name' if 'name' in df.columns else 'structname'
    df = df[['index', name_col]].rename(columns={name_col: 'name'})

    out_file = stats_file.with_name('brainstem_dseg.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)


def brainstem_stats_to_stats(subjects_dir: str, subject_id: str) -> str:
    """Generate a TSV table of morphological statistics from ``brainstem.v13.stats``."""
    from pathlib import Path

    from petprep.utils.gtmseg import _read_stats_table
    stats_file = Path(subjects_dir) / subject_id / 'stats' / 'brainstem.v13.stats'
    df = _read_stats_table(stats_file)
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

    out_file = stats_file.with_name('brainstem_morph.tsv')
    df.to_csv(out_file, sep='\t', index=False)
    return str(out_file)


def _read_volumes_table(lut_file: str | Path) -> pd.DataFrame:
    """Parse a brainstem LUT/volumes table.

    Parameters
    ----------
    lut_file : :class:`str` or :class:`~pathlib.Path`
        Path to the LUT/volumes file produced by ``segmentBS.sh``.

    Returns
    -------
    :class:`pandas.DataFrame`
        Table with ``index`` and ``name`` columns. If a numeric code is present
        as the second column, the name will be read from the third column.
    """
    from pathlib import Path
    import pandas as pd

    lut_file = Path(lut_file)
    rows: list[dict[str, str]] = []
    with lut_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            idx = int(tokens[0])
            if len(tokens) > 2 and tokens[1].lstrip('-').isdigit():
                name = tokens[2]
            else:
                name = tokens[1]
            rows.append({'index': idx, 'name': name})

    return pd.DataFrame(rows)