import nibabel as nb
import numpy as np
import pytest
from pathlib import Path
from nipype.interfaces.base import Undefined

from .... import config
from ...tests import mock_config
from ..segmentation import _merge_ha_labels, init_segmentation_wf


def test_segmentation_node_selection(tmp_path):
    """Ensure workflow nodes depend on segmentation type."""
    with mock_config():
        config.workflow.tpl_file = 'tpl.nii.gz'
        config.workflow.atlas_file = 'atlas_dseg.nii.gz'
        config.workflow.seg = 'atlas'

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

        wf_atlas = init_segmentation_wf('atlas')
        names_atlas = [n.name for n in wf_atlas._get_all_nodes()]
        assert 'tpl_source' in names_atlas
        assert 'atlas_source' in names_atlas
        assert 'warp_atlas' in names_atlas
        assert 'warp_tpl' in names_atlas
        assert 'ds_tpl_t1w' in names_atlas
        assert 'ds_atlasseg' in names_atlas
        assert 'ds_atlasdsegtsv' in names_atlas
        assert 'ds_atlasmorphtsv' in names_atlas
        assert 'segstats_atlas' in names_atlas

        ds_tpl = wf_atlas.get_node('ds_tpl_t1w')
        assert ds_tpl.inputs.desc == 'tpl'
        assert ds_tpl.inputs.suffix == 'T1w'

        ds_seg = wf_atlas.get_node('ds_atlasseg')
        assert ds_seg.inputs.seg == config.workflow.seg


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


def test_atlas_label_connections():
    """Atlas label table should propagate to TSV builders."""
    with mock_config():
        config.workflow.tpl_file = 'tpl.nii.gz'
        config.workflow.atlas_file = 'atlas_dseg.nii.gz'

        wf = init_segmentation_wf('atlas')
        atlas_source = wf.get_node('atlas_source')
        segstats = wf.get_node('segstats_atlas')
        create_dseg = wf.get_node('create_atlas_dsegtsv')
        create_morph = wf.get_node('create_atlas_morphtsv')
        convert_seg = wf.get_node('convert_atlasseg')

        edge_ctab = wf._graph.get_edge_data(atlas_source, segstats)
        edge_seg = wf._graph.get_edge_data(convert_seg, segstats)
        edge_dseg = wf._graph.get_edge_data(atlas_source, create_dseg)
        edge_morph = wf._graph.get_edge_data(segstats, create_morph)

        assert ('labels_file', 'color_table_file') in edge_ctab['connect']
        assert ('out_file', 'segmentation_file') in edge_seg['connect']
        assert ('labels_file', 'seg_file') in edge_dseg['connect']
        assert ('summary_file', 'summary_file') in edge_morph['connect']


def test_atlas_file_path():
    """Path inputs for atlas_file are accepted."""
    with mock_config():
        config.workflow.tpl_file = 'tpl.nii.gz'
        config.workflow.atlas_file = Path('atlas_dseg.nii.gz')

        wf = init_segmentation_wf('atlas')
        atlas_source = wf.get_node('atlas_source')

        assert isinstance(atlas_source.inputs.labels_file, str)
        assert atlas_source.inputs.labels_file.endswith('atlas_dseg.tsv')


def test_atlas_warp_transforms():
    """Warp nodes should have transforms defined after initialization."""
    with mock_config():
        config.workflow.tpl_file = 'tpl.nii.gz'
        config.workflow.atlas_file = 'atlas_dseg.nii.gz'

        wf = init_segmentation_wf('atlas')
        warp_atlas = wf.get_node('warp_atlas')
        warp_tpl = wf.get_node('warp_tpl')

        assert warp_atlas.inputs.transforms is not Undefined
        assert warp_tpl.inputs.transforms is not Undefined


def test_atlas_custom_name_outputs():
    """Custom atlas name should be reflected in output node names."""
    with mock_config():
        config.workflow.tpl_file = 'tpl.nii.gz'
        config.workflow.atlas_file = 'atlas_dseg.nii.gz'
        config.workflow.atlas_name = 'custom'

        wf = init_segmentation_wf('atlas')
        names = [n.name for n in wf._get_all_nodes()]

        assert 'ds_customseg' in names
        assert 'create_custom_dsegtsv' in names
        assert 'create_custom_morphtsv' in names
