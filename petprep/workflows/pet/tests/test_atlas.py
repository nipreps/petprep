import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from importlib.resources import files as ir_files

from petprep import config
from ..atlas import init_atlas_wf, _atlas_morph_tsv


def test_init_atlas_wf_build(tmp_path, monkeypatch):
    t1_img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    t1_file = tmp_path / 't1w.nii.gz'
    t1_img.to_filename(t1_file)

    atlas_img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    atlas_file = tmp_path / 'atlas.nii.gz'
    atlas_img.to_filename(atlas_file)

    labels_file = tmp_path / 'labels.tsv'
    labels_file.write_text('index\tname\n0\tbackground\n')

    cfg_file = ir_files('petprep.data.atlas') / 'config.json'

    def fake_get(**kwargs):
        if kwargs.get('suffix') == 'T1w':
            return str(t1_file)
        if kwargs.get('extension') == 'tsv':
            return str(labels_file)
        return str(atlas_file)

    monkeypatch.setattr('petprep.workflows.pet.atlas.get_template', fake_get)

    config.execution.petprep_dir = tmp_path
    config.execution.dataset_links = {}

    wf = init_atlas_wf(
        atlas='MIAL67ThalamicNuclei',
        config_file=str(cfg_file),
        tpl2anat_xfm=None,
    )
    node_names = [n.name for n in wf._get_all_nodes()]
    assert 'apply_atlas' in node_names
    assert 't1_to_tpl' in node_names
    assert wf.get_node('label_source').inputs.dseg_tsv == str(labels_file)
    assert wf.get_node('ds_seg').inputs.seg == 'MIAL67ThalamicNuclei'
    assert wf.get_node('ds_dseg_tsv').inputs.seg == 'MIAL67ThalamicNuclei'
    assert wf.get_node('ds_morph_tsv').inputs.seg == 'MIAL67ThalamicNuclei'


def test_init_atlas_wf_with_xfm(tmp_path, monkeypatch):
    t1_img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    t1_file = tmp_path / 't1w.nii.gz'
    t1_img.to_filename(t1_file)

    atlas_img = nb.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    atlas_file = tmp_path / 'atlas.nii.gz'
    atlas_img.to_filename(atlas_file)

    labels_file = tmp_path / 'labels.tsv'
    labels_file.write_text('index\tname\n0\tbackground\n')

    cfg_file = ir_files('petprep.data.atlas') / 'config.json'

    def fake_get(**kwargs):
        if kwargs.get('suffix') == 'T1w':
            return str(t1_file)
        if kwargs.get('extension') == 'tsv':
            return str(labels_file)
        return str(atlas_file)

    monkeypatch.setattr('petprep.workflows.pet.atlas.get_template', fake_get)

    config.execution.petprep_dir = tmp_path
    config.execution.dataset_links = {}

    xfm = tmp_path / 'tpl2anat.txt'
    xfm.write_text('0')

    wf = init_atlas_wf(
        atlas='MIAL67ThalamicNuclei',
        config_file=str(cfg_file),
        tpl2anat_xfm=None,
    )

    t1_to_tpl = wf.get_node('t1_to_tpl')
    t1_to_tpl.inputs.tpl2anat_xfm = str(xfm)
    t1_to_tpl.inputs.t1w_preproc = str(t1_file)

    def _fail_run(self, *args, **kwargs):
        raise AssertionError('Registration should not run')

    monkeypatch.setattr('petprep.workflows.pet.atlas.Registration.run', _fail_run)
    result = t1_to_tpl.run()
    assert result.outputs.tpl2anat_xfm == str(xfm)


def test_init_atlas_wf_bad_name(tmp_path):
    cfg_file = ir_files('petprep.data.atlas') / 'config.json'
    with pytest.raises(ValueError, match="not found"):
        init_atlas_wf(atlas='notreal', config_file=str(cfg_file), tpl2anat_xfm=None)


def test_atlas_morph_tsv(tmp_path):
    data = np.array([
        [[0, 1], [1, 1]],
        [[0, 0], [1, 0]],
    ], dtype='int16')
    seg = nb.Nifti1Image(data, np.eye(4))
    seg_file = tmp_path / 'seg.nii.gz'
    seg.to_filename(seg_file)

    labels = tmp_path / 'labels.tsv'
    labels.write_text('index\tname\n0\tbg\n1\tregion\n')

    out = _atlas_morph_tsv(str(seg_file), str(labels))
    df = pd.read_csv(out, sep='\t')
    volume = df.loc[df['index'] == 1, 'volume-mm3'].iloc[0]
    assert volume == 4
