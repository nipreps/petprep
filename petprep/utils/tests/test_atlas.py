from __future__ import annotations

from pathlib import Path

import nibabel as nb
import numpy as np

from petprep.utils import atlas as atlas_utils


def test_get_atlas_files_fallback(tmp_path, monkeypatch):
    src_dir = tmp_path / 'source'
    src_dir.mkdir()
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    invalid_seg = src_dir / 'invalid.nii.gz'
    invalid_seg.write_text('not a nifti')

    valid_seg = src_dir / 'valid.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)).to_filename(valid_seg)

    labels = src_dir / 'atlas.tsv'
    labels.write_text('index\tname\n1\tTest\n')

    monkeypatch.setattr(
        atlas_utils,
        'load_atlas_config',
        lambda: {
            'test': {
                'template': 'dummy',
                'segmentation': [
                    {'source': 'file', 'path': str(invalid_seg)},
                    {'source': 'file', 'path': str(valid_seg)},
                ],
                'labels': {'source': 'file', 'path': str(labels)},
            }
        },
    )

    monkeypatch.chdir(work_dir)
    seg_file, label_file = atlas_utils.get_atlas_files('test')

    seg_file = Path(seg_file)
    label_file = Path(label_file)

    assert seg_file.parent == work_dir
    assert seg_file.name == valid_seg.name
    assert seg_file.read_bytes() == valid_seg.read_bytes()

    assert label_file.parent == work_dir
    assert label_file.name == labels.name
    assert label_file.read_text() == labels.read_text()
