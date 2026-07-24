import sys
import types

import nibabel as nb
import numpy as np
import pytest

from petprep.utils import atlas


def test_load_atlas_config_contains_known_atlas():
    atlas.load_atlas_config.cache_clear()
    config = atlas.load_atlas_config()
    assert 'HOCPA' in config
    assert 'segmentation' in config['HOCPA']


def test_resolve_resource_templateflow(monkeypatch):
    tf_api = types.SimpleNamespace(get=lambda **kwargs: '/tmp/templateflow.nii.gz')
    tf_module = types.SimpleNamespace(api=tf_api)
    monkeypatch.setitem(sys.modules, 'templateflow', tf_module)
    monkeypatch.setitem(sys.modules, 'templateflow.api', tf_api)

    resource = {'source': 'templateflow', 'query': {'atlas': 'HOCPA'}}
    resolved = atlas._resolve_resource('MNI152NLin6Asym', resource)
    assert resolved == '/tmp/templateflow.nii.gz'


def test_resolve_resource_templateflow_empty(monkeypatch):
    tf_api = types.SimpleNamespace(get=lambda **kwargs: [])
    tf_module = types.SimpleNamespace(api=tf_api)
    monkeypatch.setitem(sys.modules, 'templateflow', tf_module)
    monkeypatch.setitem(sys.modules, 'templateflow.api', tf_api)

    resource = {'source': 'templateflow', 'query': {'atlas': 'HOCPA'}}
    with pytest.raises(ValueError, match='No files found for atlas resource'):
        atlas._resolve_resource('MNI152NLin6Asym', resource)


def test_resolve_resource_package_and_file(tmp_path):
    resource_package = {'source': 'package', 'path': 'segmentation/brainstem.txt'}
    resolved_package = atlas._resolve_resource('MNI152NLin6Asym', resource_package)
    assert resolved_package.endswith('petprep/data/segmentation/brainstem.txt')

    file_path = tmp_path / 'atlas_labels.tsv'
    file_path.write_text('index\tname\n1\tone\n')
    resource_file = {'source': 'file', 'path': str(file_path)}
    resolved_file = atlas._resolve_resource('MNI152NLin6Asym', resource_file)
    assert resolved_file == str(file_path)


def test_get_atlas_files_success(monkeypatch, tmp_path):
    seg_file = tmp_path / 'seg.nii.gz'
    labels_file = tmp_path / 'labels.tsv'
    nb.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine=np.eye(4)).to_filename(seg_file)
    labels_file.write_text('index\tname\n1\tone\n')

    config = {
        'Demo': {
            'template': 'MNI152NLin6Asym',
            'segmentation': {'source': 'file', 'path': str(seg_file)},
            'labels': {'source': 'file', 'path': str(labels_file)},
        }
    }

    monkeypatch.setattr(atlas, 'load_atlas_config', lambda: config)
    monkeypatch.setattr(atlas, '_materialize_resource', lambda path: path)

    seg, labels = atlas.get_atlas_files('Demo')
    assert seg == str(seg_file)
    assert labels == str(labels_file)


def test_get_atlas_files_missing_entries(monkeypatch):
    monkeypatch.setattr(atlas, 'load_atlas_config', lambda: {'Empty': {'template': 'MNI'}})
    with pytest.raises(ValueError, match='must define both'):
        atlas.get_atlas_files('Empty')


def test_get_atlas_files_unknown_atlas(monkeypatch):
    monkeypatch.setattr(atlas, 'load_atlas_config', dict)
    with pytest.raises(ValueError, match='is not defined'):
        atlas.get_atlas_files('Missing')
