from pathlib import Path

from petprep.utils.brainstem import _read_volumes_table


def test_lut_parses_standard(tmp_path: Path):
    lut_file = tmp_path / 'lut.txt'
    lut_file.write_text('1 RegionA 1 2 3\n2 RegionB 4 5 6\n')

    df = _read_volumes_table(lut_file)

    assert list(df['name']) == ['RegionA', 'RegionB']


def test_lut_parses_second_numeric(tmp_path: Path):
    lut_file = tmp_path / 'lut.txt'
    lut_file.write_text('1 101 RegionA 1 2 3\n2 102 RegionB 4 5 6\n')

    df = _read_volumes_table(lut_file)

    assert list(df['name']) == ['RegionA', 'RegionB']