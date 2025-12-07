import nibabel as nb
import numpy as np
import pandas as pd

from ..segmentation import (
    _read_stats_table,
    atlas_segmentation_to_morph,
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


def test_ctab_to_dsegtsv(tmp_path):
    ctab = tmp_path / 'test.ctab'
    ctab.write_text("""1 one 0 0 0 0\n2 two 0 0 0 0\n""")
    out = ctab_to_dsegtsv(ctab)
    df = pd.read_csv(out, sep='\t')
    assert list(df.columns) == ['index', 'name']


def test_atlas_segmentation_to_morph(tmp_path):
    seg_file = tmp_path / 'atlas_seg.nii.gz'
    data = np.array([[[1, 1], [0, 2]]], dtype=np.int16)
    img = nb.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    img.to_filename(seg_file)

    label_file = tmp_path / 'labels.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['one', 'two']}).to_csv(label_file, sep='\t', index=False)

    out, meta = atlas_segmentation_to_morph(seg_file, label_file)
    df = pd.read_csv(out, sep='\t')

    assert list(df.columns) == ['index', 'name', 'volume-mm3']
    assert df['volume-mm3'].tolist() == [2.0, 1.0]
    assert 'Columns' in meta and 'volume-mm3' in meta['Columns']
