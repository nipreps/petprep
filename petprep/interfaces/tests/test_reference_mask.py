import json
from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.pipeline import engine as pe

from ...data import load as load_data
from ..reference_mask import ExtractRefRegion


def _create_seg(tmp_path: Path) -> Path:
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[1, 1, 1] = 1
    data[2, 2, 2] = 2
    img = nb.Nifti1Image(data, np.eye(4))
    seg_file = tmp_path / 'seg.nii.gz'
    img.to_filename(seg_file)
    return seg_file


def _create_config(tmp_path: Path, indices, extra=None):
    cfg = {'testseg': {'region': {'refmask_indices': indices}}}
    if extra:
        cfg['testseg']['region'].update(extra)
    cfg_file = tmp_path / 'config.json'
    cfg_file.write_text(json.dumps(cfg))
    return cfg_file


def test_extract_refregion(tmp_path):
    seg = _create_seg(tmp_path)
    cfg = _create_config(tmp_path, [1])

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            config_file=str(cfg),
            segmentation_type='testseg',
            region_name='region',
        ),
        name='er',
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out[1, 1, 1] == 1
    assert out.sum() == 1


def test_extract_refregion_override(tmp_path):
    seg = _create_seg(tmp_path)
    cfg = _create_config(tmp_path, [1])

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            config_file=str(cfg),
            segmentation_type='testseg',
            region_name='region',
            override_indices=[2],
        ),
        name='er2',
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out[2, 2, 2] == 1
    assert out.sum() == 1


def test_extract_refregion_override_ignores_config(tmp_path):
    seg = _create_seg(tmp_path)
    cfg = _create_config(
        tmp_path,
        [1],
        {
            'exclude_indices': [2],
            'erode_by_voxels': 1,
        },
    )

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            config_file=str(cfg),
            segmentation_type='testseg',
            region_name='region',
            override_indices=[2],
        ),
        name='er3',
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out[2, 2, 2] == 1
    assert out.sum() == 1


def test_extract_refregion_override_missing_config(tmp_path):
    seg = _create_seg(tmp_path)
    cfg_file = tmp_path / 'config.json'
    cfg_file.write_text(json.dumps({'testseg': {}}))

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            config_file=str(cfg_file),
            segmentation_type='testseg',
            region_name='missing',
            override_indices=[2],
        ),
        name='er4',
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out[2, 2, 2] == 1
    assert out.sum() == 1


def test_extract_refregion_gm_threshold(tmp_path):
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[1, 1, 1] = 1
    data[2, 2, 2] = 1
    seg = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(seg)
    gm = np.zeros((5, 5, 5), dtype=np.float32)
    gm[1, 1, 1] = 0.4
    gm[2, 2, 2] = 0.8
    gm_file = tmp_path / 'gm.nii.gz'
    nb.Nifti1Image(gm, np.eye(4)).to_filename(gm_file)

    cfg = _create_config(tmp_path, [1], {'gm_prob_threshold': 0.5})

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg),
            gm_probseg=str(gm_file),
            config_file=str(cfg),
            segmentation_type='testseg',
            region_name='region',
        ),
        name='er5',
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()
    assert out.sum() == 1
    assert out[2, 2, 2] == 1


def test_extract_refregion_cc_from_aparcaseg_config(tmp_path):
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[1, 1, 1] = 251
    data[1, 1, 2] = 252
    data[1, 1, 3] = 253
    data[1, 2, 1] = 254
    data[1, 2, 2] = 255
    data[2, 2, 2] = 250
    seg_file = tmp_path / 'aparc+aseg.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(seg_file)

    node = pe.Node(
        ExtractRefRegion(
            seg_file=str(seg_file),
            config_file=str(load_data('reference_mask/config.json')),
            segmentation_type='aparcaseg',
            region_name='cc',
        ),
        name='er_cc',
        base_dir=str(tmp_path),
    )
    res = node.run()
    out = nb.load(res.outputs.refmask_file).get_fdata()

    assert out.sum() == 5
    assert out[2, 2, 2] == 0
