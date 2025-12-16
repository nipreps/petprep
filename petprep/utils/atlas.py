"""Helpers for template-driven atlas segmentations."""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4

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


def _materialize_resource(resource_path: str) -> str:
    """Copy a resolved resource into the current working directory."""

    src = Path(resource_path)
    try:
        src_parent = src.resolve().parent
        cwd = Path.cwd().resolve()
    except Exception:  # pragma: no cover - best-effort resolution
        src_parent = src.parent
        cwd = Path.cwd()

    if src_parent == cwd:
        return str(src)

    suffix = ''.join(src.suffixes)
    base = src.name[: -len(suffix)] if suffix else src.name
    dest = cwd / src.name

    if dest == src:
        return str(dest)

    if dest.exists():
        try:
            if dest.resolve() == src.resolve():
                return str(dest)
        except Exception:  # pragma: no cover - fallback to bytes comparison
            if dest.read_bytes() == src.read_bytes():
                return str(dest)
        dest = cwd / f'{base}_{uuid4().hex}{suffix}'

    shutil.copy2(src, dest)
    return str(dest)


def get_atlas_files(atlas_name: str) -> tuple[str, str]:
    """Return the segmentation and label files for a configured atlas."""

    # nipype's ``Function`` interface may serialize this function into a fresh
    # namespace that lacks the module-level globals, so import the loader here
    # to guarantee availability when executed in a worker process.
    from petprep.utils.atlas import (
        _materialize_resource,
        _resolve_resource,
        load_atlas_config,
    )

    atlas_config = load_atlas_config().get(atlas_name)
    if atlas_config is None:
        raise ValueError(f"Atlas '{atlas_name}' is not defined in the atlas configuration file")

    segmentation = atlas_config.get('segmentation')
    labels = atlas_config.get('labels')

    if not segmentation or not labels:
        raise ValueError(
            f"Atlas '{atlas_name}' must define both 'segmentation' and 'labels' entries in the configuration"
        )

    def _validate_nifti(image_path: str) -> None:
        """Ensure an atlas NIfTI image can be loaded."""
        import nibabel as nb

        try:
            nb.load(image_path)
        except Exception as exc:  # pragma: no cover - sanity check
            raise ValueError(f'Atlas file {image_path} is not a valid NIfTI image') from exc

    template = atlas_config['template']
    segmentation_resources = segmentation if isinstance(segmentation, (list, tuple)) else [segmentation]
    seg_file = None
    seg_errors: list[str] = []

    for seg_resource in segmentation_resources:
        try:
            seg_candidate = _resolve_resource(template, seg_resource)
            _validate_nifti(seg_candidate)
            seg_candidate = _materialize_resource(seg_candidate)
        except ValueError as exc:
            seg_errors.append(str(exc))
            seg_candidate = None

        if seg_candidate:
            seg_file = seg_candidate
            break

    if seg_file is None:
        error_msg = '; '.join(seg_errors) or f'No valid segmentation resources for atlas {atlas_name}'
        raise ValueError(error_msg)

    label_file = _materialize_resource(_resolve_resource(template, labels))
    return seg_file, label_file
