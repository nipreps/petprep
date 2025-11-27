"""Helpers for template-driven atlas segmentations."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from importlib.resources import files as ir_files

from petprep.data import load as load_data


@lru_cache
def load_atlas_config() -> dict[str, Any]:
    """Load atlas configuration bundled with *PETPrep*.

    The configuration maps atlas names to metadata describing the template,
    segmentation image and corresponding label table. Both files can be
    referenced as package data or retrieved from TemplateFlow.
    """

    config_file = ir_files('petprep.data.segmentation') / 'atlases.json'
    return json.loads(config_file.read_text())


def _resolve_resource(template: str, resource: dict[str, Any]) -> str:
    """Resolve a single atlas resource to a filesystem path."""

    source = resource.get('source', 'templateflow')
    if source == 'templateflow':
        import templateflow.api as tf

        query = {**resource.get('query', {}), 'template': template}
        result = tf.get(**query)
        if isinstance(result, (list, tuple)):
            if not result:
                raise ValueError(f'No files found for atlas resource: {resource}')
            result = result[0]
        return str(result)

    if source == 'package':
        return str(load_data(resource['path']))

    if source == 'file':
        return str(Path(resource['path']).absolute())

    raise ValueError(f"Unsupported atlas source '{source}'")


def get_atlas_files(atlas_name: str) -> tuple[str, str]:
    """Return the segmentation and label files for a configured atlas."""

    # nipype's ``Function`` interface may serialize this function into a fresh
    # namespace that lacks the module-level globals, so import the loader here
    # to guarantee availability when executed in a worker process.
    from petprep.utils.atlas import _resolve_resource, load_atlas_config

    atlas_config = load_atlas_config().get(atlas_name)
    if atlas_config is None:
        raise ValueError(f"Atlas '{atlas_name}' is not defined in the atlas configuration file")

    segmentation = atlas_config.get('segmentation')
    labels = atlas_config.get('labels')

    if not segmentation or not labels:
        raise ValueError(
            f"Atlas '{atlas_name}' must define both 'segmentation' and 'labels' entries in the configuration"
        )

    seg_file = _resolve_resource(atlas_config['template'], segmentation)
    label_file = _resolve_resource(atlas_config['template'], labels)
    return seg_file, label_file
