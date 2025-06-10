from pathlib import Path
import pandas as pd

from petprep.utils.thalamic import summary_to_stats, ctab_to_dsegtsv


def test_thalamic_utils(tmp_path: Path):
    summary_file = tmp_path / 'summary.stats'
    summary_file.write_text(
        '# ColHeaders Index SegId Name Volume_mm3\n'
        '1 10 region1 5\n'
        '2 11 region2 7\n'
    )

    out_stats = summary_to_stats(summary_file)
    df_stats = pd.read_csv(out_stats, sep='\t')
    assert list(df_stats.columns) == ['index', 'name', 'volume-mm3']

    ctab_file = tmp_path / 'labels.ctab'
    ctab_file.write_text('1 region1 0 0 0 0\n2 region2 0 0 0 0\n')
    out_dseg = ctab_to_dsegtsv(ctab_file)
    df_dseg = pd.read_csv(out_dseg, sep='\t')
    assert list(df_dseg.columns) == ['index', 'name']