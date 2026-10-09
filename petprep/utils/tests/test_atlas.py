import sys
import types
from importlib.resources import files
from pathlib import Path

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


@pytest.mark.parametrize('networks', [7, 17])
def test_get_buckner_atlas_files(monkeypatch, tmp_path, networks):
    """Resolve each network variant and its labels using TemplateFlow entities."""
    import templateflow.api as tf
    from bids import BIDSLayout

    template_dir = tmp_path / 'templateflow' / 'tpl-MNI152NLin6Asym'
    template_dir.mkdir(parents=True)
    for variant in (7, 17):
        for resolution in (1, 2):
            stem = (
                f'tpl-MNI152NLin6Asym_res-{resolution:02d}_atlas-Buckner2011_seg-{variant}n_dseg'
            )
            nb.Nifti1Image(
                np.full((2, 2, 2), variant, dtype=np.uint8), affine=np.eye(4)
            ).to_filename(template_dir / f'{stem}.nii.gz')
            (template_dir / f'{stem}.tsv').write_text(
                f'index\tname\n{variant}\tNetwork_{variant}\n'
            )

    layout = BIDSLayout(
        template_dir.parent,
        validate=False,
        config=str(files('templateflow') / 'conf' / 'config.json'),
    )
    monkeypatch.setattr(tf, 'get', lambda **query: layout.get(return_type='file', **query))
    monkeypatch.chdir(tmp_path)

    seg, labels = atlas.get_atlas_files(f'Buckner2011{networks}Networks')

    expected_stem = f'tpl-MNI152NLin6Asym_res-01_atlas-Buckner2011_seg-{networks}n_dseg'
    assert Path(seg) == tmp_path / f'{expected_stem}.nii.gz'
    assert Path(labels) == tmp_path / f'{expected_stem}.tsv'
    assert np.all(nb.load(seg).get_fdata() == networks)
    assert Path(labels).read_text() == f'index\tname\n{networks}\tNetwork_{networks}\n'


def test_get_atlas_files_missing_entries(monkeypatch):
    monkeypatch.setattr(atlas, 'load_atlas_config', lambda: {'Empty': {'template': 'MNI'}})
    with pytest.raises(ValueError, match='must define both'):
        atlas.get_atlas_files('Empty')


def test_get_atlas_files_unknown_atlas(monkeypatch):
    monkeypatch.setattr(atlas, 'load_atlas_config', dict)
    with pytest.raises(ValueError, match='is not defined'):
        atlas.get_atlas_files('Missing')
