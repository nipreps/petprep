from __future__ import annotations

from pathlib import Path

from .gtmseg import _read_stats_table


def brainstem_to_dsegtsv(subjects_dir: str, subject_id: str) -> str:
    """Generate a TSV table describing brainstem segmentation labels."""
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