from pathlib import Path

import nibabel as nb
import numpy as np
import pytest

from petprep.interfaces.motion import MotionPlot


def _write_image(path: Path, shape):
    data = np.linspace(0, 1, int(np.prod(shape)), dtype=float).reshape(shape)
    img = nb.Nifti1Image(data, np.eye(4))
    img.to_filename(path)
    return path


def test_motion_plot_builds_svg(tmp_path, monkeypatch):
    orig_path = _write_image(tmp_path / 'orig.nii.gz', (4, 4, 4, 2))
    corr_path = _write_image(tmp_path / 'corr.nii.gz', (4, 4, 4, 2))

    call_count = {'count': 0}

    def fake_plot_epi(img, **kwargs):
        height = 10 if call_count['count'] % 2 == 0 else 6
        array = np.ones((height, 8, 3), dtype=np.uint8) * 255
        from imageio import v2 as imageio

        imageio.imwrite(kwargs['output_file'], array)
        call_count['count'] += 1

    monkeypatch.setattr('petprep.interfaces.motion.plot_epi', fake_plot_epi)

    motion = MotionPlot()
    motion.inputs.original_pet = str(orig_path)
    motion.inputs.corrected_pet = str(corr_path)
    motion.inputs.duration = 0.05

    result = motion.run(cwd=tmp_path)
    svg_file = Path(result.outputs.svg_file)

    content = svg_file.read_text()
    assert 'frame-0' in content
    assert 'animation-delay: 0.05s' in content
    assert call_count['count'] == 4


def test_compute_display_params_handles_single_frame(tmp_path):
    img_path = _write_image(tmp_path / 'single.nii.gz', (5, 5, 5))

    motion = MotionPlot()
    mid_img, cut_coords, vmin, vmax = motion._compute_display_params(str(img_path))

    assert mid_img.ndim == 3
    assert len(cut_coords) == 3
    assert vmin <= vmax


def test_load_framewise_displacement_variants(tmp_path):
    fd_path = tmp_path / 'fd.tsv'
    fd_path.write_text('framewise_displacement\n0.1\n0.2\n')

    motion = MotionPlot()
    values = motion._load_framewise_displacement(str(fd_path))
    assert np.allclose(values, [0.1, 0.2])

    fd_path.write_text('FD\n0.0\n')
    values = motion._load_framewise_displacement(str(fd_path))
    assert np.allclose(values, [0.0])

    fd_path.write_text('other\n1.0\n')
    with pytest.raises(ValueError):
        motion._load_framewise_displacement(str(fd_path))


def test_build_animation_includes_fd_plot(tmp_path, monkeypatch):
    orig_path = _write_image(tmp_path / 'orig.nii.gz', (4, 4, 4, 3))
    corr_path = _write_image(tmp_path / 'corr.nii.gz', (4, 4, 4, 3))
    fd_path = tmp_path / 'fd.tsv'
    fd_path.write_text('FD\n0\n0\n')

    def fake_plot_epi(img, **kwargs):
        array = np.ones((8, 8, 3), dtype=np.uint8) * 255
        from imageio import v2 as imageio

        imageio.imwrite(kwargs['output_file'], array)

    monkeypatch.setattr('petprep.interfaces.motion.plot_epi', fake_plot_epi)

    motion = MotionPlot()
    motion.inputs.original_pet = str(orig_path)
    motion.inputs.corrected_pet = str(corr_path)
    motion.inputs.fd_file = str(fd_path)
    motion.inputs.duration = 0.01

    result = motion.run(cwd=tmp_path)
    svg_file = Path(result.outputs.svg_file)
    content = svg_file.read_text()

    assert 'fd-plot' in content
    assert 'FD (mm)' in content
    assert 'frame-2' not in content  # limited to FD length
