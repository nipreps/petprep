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

        wf_cc = init_segmentation_wf('cc')
        names_cc = [n.name for n in wf_cc._get_all_nodes()]
        assert 'run_cc' in names_cc
        assert 'convert_ccseg' in names_cc
        assert 'segstats_cc' in names_cc


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


def test_template_atlas_masking():
    """Template atlas workflows should optionally mask warped segmentations."""
    with mock_config():
        wf = init_segmentation_wf('HOCPA')

        names = [n.name for n in wf._get_all_nodes()]
        assert 'mask_HOCPA_atlas' in names

        apply_node = wf.get_node('warp_HOCPA_atlas')
        mask_node = wf.get_node('mask_HOCPA_atlas')
        seg_source = wf.get_node('HOCPA_seg_source')
        inputnode = wf.get_node('inputnode')

        edge_apply = wf._graph.get_edge_data(apply_node, mask_node)
        edge_mask = wf._graph.get_edge_data(inputnode, mask_node)
        edge_seg = wf._graph.get_edge_data(mask_node, seg_source)

        assert ('output_image', 'in_file') in edge_apply['connect']
        assert ('anat_ribbon', 'in_mask') in edge_mask['connect']
        assert ('out_file', 'segmentation') in edge_seg['connect']


def test_template_atlas_masking_unsupported_option():
    """Atlas masking should only allow brain or ribbon choices."""
    from copy import deepcopy

    from .. import segmentation

    bad_spec = deepcopy(segmentation.SEGMENTATIONS['HOCPA'])
    bad_spec['template_atlas'] = deepcopy(bad_spec['template_atlas'])
    bad_spec['template_atlas']['mask'] = 'cortex'

    segmentation.SEGMENTATIONS['HOCPA_bad'] = bad_spec

    with mock_config(), pytest.raises(ValueError):
        init_segmentation_wf('HOCPA_bad')

    segmentation.SEGMENTATIONS.pop('HOCPA_bad', None)
