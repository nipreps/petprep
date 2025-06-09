from pathlib import Path

from nipype.interfaces import utility as niu

from ....utils.brainstem import brainstem_stats_to_stats, brainstem_to_dsegtsv


def _make_stats_file(base: Path) -> Path:
    stats_file = base / 'sub-01' / 'stats' / 'brainstem.v13.stats'
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(
        '# Dummy stats\n# ColHeaders Index SegId Name Volume_mm3\n1 2 region1 10\n2 4 region2 20\n'
    )
    return stats_file


def test_brainstem_functions_via_niu(tmp_path: Path):
    subjects_dir = tmp_path
    _make_stats_file(subjects_dir)

    node = niu.Function(
        function=brainstem_to_dsegtsv,
        input_names=['subjects_dir', 'subject_id'],
        output_names=['out_file'],
    )
    node.base_dir = tmp_path
    node.inputs.subjects_dir = str(subjects_dir)
    node.inputs.subject_id = 'sub-01'
    res = node.run()
    out_file = Path(res.outputs.out_file)
    assert out_file.exists()
    header = out_file.read_text().splitlines()[0]
    assert header == 'index\tname'

    node2 = niu.Function(
        function=brainstem_stats_to_stats,
        input_names=['subjects_dir', 'subject_id'],
        output_names=['out_file'],
    )
    node2.base_dir = tmp_path
    node2.inputs.subjects_dir = str(subjects_dir)
    node2.inputs.subject_id = 'sub-01'
    res2 = node2.run()
    out_file2 = Path(res2.outputs.out_file)
    assert out_file2.exists()
    header2 = out_file2.read_text().splitlines()[0]
    assert header2 == 'index\tname\tvolume-mm3'