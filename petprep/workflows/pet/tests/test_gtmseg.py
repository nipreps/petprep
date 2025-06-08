from pathlib import Path

from nipype.interfaces import utility as niu

from ....utils.gtmseg import gtm_stats_to_stats, gtm_to_dsegtsv


def _make_stats_file(base: Path) -> Path:
    stats_file = base / 'sub-01' / 'stats' / 'gtmseg.stats'
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(
        '# Dummy stats\n'
        '# ColHeaders Index SegId Name Volume_mm3\n'
        '1 2 region1 10\n'
        '2 4 region2 20\n'
    )
    return stats_file


def test_gtmseg_functions_via_niu(tmp_path: Path):
    subjects_dir = tmp_path
    _make_stats_file(subjects_dir)

    node = niu.Function(
        function=gtm_to_dsegtsv,
        input_names=['subjects_dir', 'subject_id'],
        output_names=['out_file'],
    )
    node.base_dir = tmp_path
    node.inputs.subjects_dir = str(subjects_dir)
    node.inputs.subject_id = 'sub-01'
    res = node.run()
    out_file = Path(res.outputs.out_file)
    assert out_file.exists()
    header1 = out_file.read_text().splitlines()[0]
    assert header1 == 'index\tname'

    node2 = niu.Function(
        function=gtm_stats_to_stats,
        input_names=['subjects_dir', 'subject_id'],
        output_names=['out_file'],
    )
    node2.base_dir = tmp_path
    node2.inputs.subjects_dir = str(subjects_dir)
    node2.inputs.subject_id = 'sub-01'
    res2 = node2.run()
    out_file2 = Path(res2.outputs.out_file)
    assert out_file2.exists()
    header = out_file2.read_text().splitlines()[0]
    assert header == 'index\tname\tvolume-mm3'
