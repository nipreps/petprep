import nibabel as nb
import numpy as np
import pytest

from ...tests import mock_config
from ..segmentation import _merge_ha_labels, init_segmentation_wf


def test_segmentation_node_selection():
    """Ensure workflow nodes depend on segmentation type."""
    with mock_config():
        wf_gtm = init_segmentation_wf('gtm')
        names_gtm = [n.name for n in wf_gtm._get_all_nodes()]
        assert 'make_gtmdsegtsv' in names_gtm
        assert 'make_gtmmorphtsv' in names_gtm
        assert 'segstats_gtm' not in names_gtm

        wf_wm = init_segmentation_wf('wm')
        names_wm = [n.name for n in wf_wm._get_all_nodes()]
        assert 'segstats_wm' in names_wm
        assert 'create_wm_dsegtsv' in names_wm
        assert 'create_wm_morphtsv' in names_wm


def test_merge_ha_labels(tmp_path):
    """Merged volume should match input geometry."""
    shape = (5, 5, 5)
    affine = np.eye(4)
    lh_data = np.zeros(shape, dtype=np.int16)
    rh_data = np.ones(shape, dtype=np.int16)

    lh_file = tmp_path / 'lh.nii.gz'
    rh_file = tmp_path / 'rh.nii.gz'
    nb.Nifti1Image(lh_data, affine).to_filename(lh_file)
    nb.Nifti1Image(rh_data, affine).to_filename(rh_file)

    out_file = _merge_ha_labels(str(lh_file), str(rh_file))
    out_img = nb.load(out_file)
    assert out_img.shape == shape
    assert np.allclose(out_img.affine, affine)
    assert np.array_equal(out_img.get_fdata().astype(np.int16), rh_data)


def test_merge_ha_labels_misaligned(tmp_path):
    """Mismatched inputs should raise a ValueError."""
    lh_file = tmp_path / 'lh.nii.gz'
    rh_file = tmp_path / 'rh.nii.gz'
    nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)).to_filename(lh_file)
    nb.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)).to_filename(rh_file)

    with pytest.raises(ValueError):
        _merge_ha_labels(str(lh_file), str(rh_file))


def test_gtm_connections():
    """GTM-specific outputs should depend on segmentation output."""
    with mock_config():
        wf = init_segmentation_wf('gtm')
        seg_node = wf.get_node('run_gtm')
        make_dseg = wf.get_node('make_gtmdsegtsv')
        make_morph = wf.get_node('make_gtmmorphtsv')

        edge_dseg = wf._graph.get_edge_data(seg_node, make_dseg)
        edge_morph = wf._graph.get_edge_data(seg_node, make_morph)

        assert ('out_file', 'seg_file') in edge_dseg['connect']
        assert ('out_file', 'seg_file') in edge_morph['connect']
