import pandas as pd

from ..segmentation import (
    _read_stats_table,
    ctab_to_dsegtsv,
    gtm_stats_to_stats,
    gtm_to_dsegtsv,
    summary_to_stats,
)


def test_read_stats_table(tmp_path):
    stats = tmp_path / 'test.stats'
    stats.write_text("""# ColHeaders Index Name Volume\n1 region1 10\n2 region2 5\n""")
    df = _read_stats_table(stats)
    assert list(df.columns) == ['Index', 'Name', 'Volume']
    assert len(df) == 2


def test_gtm_to_dsegtsv(tmp_path):
    stats_dir = tmp_path / 'sub-01' / 'stats'
    stats_dir.mkdir(parents=True)
    stats_file = stats_dir / 'gtmseg.stats'
    stats_file.write_text("""# ColHeaders Index Name Volume\n1 R1 3\n2 R2 4\n""")
    out = gtm_to_dsegtsv(tmp_path, 'sub-01')
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['index', 'name']


def test_gtm_stats_to_stats(tmp_path):
    stats_dir = tmp_path / 'sub-01' / 'stats'
    stats_dir.mkdir(parents=True)
    stats_file = stats_dir / 'gtmseg.stats'
    stats_file.write_text("""# ColHeaders Index Name Volume\n1 R1 3\n""")
    out = gtm_stats_to_stats(tmp_path, 'sub-01')
    df = pd.read_csv(out, sep='\t')
    assert 'volume-mm3' in df.columns


def test_summary_to_stats(tmp_path):
    stats = tmp_path / 'summary.stats'
    stats.write_text("""# ColHeaders Index Name Volume_mm3\n1 R1 3\n""")
    out = summary_to_stats(stats)
    df = pd.read_csv(out, sep='\t')
    assert 'volume-mm3' in df.columns
    assert (df['volume-mm3'] > 0).all()


def test_ctab_to_dsegtsv(tmp_path):
    ctab = tmp_path / 'test.ctab'
    ctab.write_text("""1 one 0 0 0 0\n2 two 0 0 0 0\n""")
    out = ctab_to_dsegtsv(ctab)
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['index', 'name']
