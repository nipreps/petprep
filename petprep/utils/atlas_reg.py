from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from ..data import load as load_data


class AtlasRegConfigError(RuntimeError):
    """Raised when atlas registration parameters cannot be loaded."""


@dataclass(frozen=True)
class AtlasRegParameterSet:
    """Container holding a resolved atlas registration parameter set."""

    atlas: str
    parameter_id: str
    description: str
    registration: Dict[str, Any]
    config_path: Path

    def as_json(self) -> str:
        """Return a JSON representation safe for serialization."""
        return json.dumps(_serialize_for_json(self.registration), indent=2, sort_keys=True)


def load_parameter_set(
    atlas: str, parameter_id: str | None = None, *, config_path: str | Path | None = None
) -> AtlasRegParameterSet:
    """
    Load the requested atlas registration parameter set.

    Parameters
    ----------
    atlas
        Atlas identifier (e.g., ``subcortex``).
    parameter_id
        Parameter set identifier. If omitted, the default entry defined in the YAML file is used.
    config_path
        Optional override path pointing to a YAML file containing the parameter definitions.
    """
    resolved_path = _resolve_config_path(atlas, config_path)
    try:
        raw_config = yaml.safe_load(resolved_path.read_text())
    except yaml.YAMLError as exc:
        raise AtlasRegConfigError(f'Could not parse YAML for atlas "{atlas}": {exc}') from exc

    if not raw_config or 'parameter_sets' not in raw_config:
        raise AtlasRegConfigError(f'Invalid atlas registration config for "{atlas}"')

    available_sets = raw_config['parameter_sets']
    if not isinstance(available_sets, dict) or not available_sets:
        raise AtlasRegConfigError(f'No parameter sets declared for atlas "{atlas}"')

    desired_id = parameter_id or raw_config.get('default')
    if desired_id is None:
        desired_id = next(iter(available_sets))

    if desired_id not in available_sets:
        available = ', '.join(sorted(available_sets))
        raise AtlasRegConfigError(
            f'Parameter set "{desired_id}" not found for atlas "{atlas}". '
            f'Available sets: {available}'
        )

    chosen = available_sets[desired_id]
    if not isinstance(chosen, dict) or 'registration' not in chosen:
        raise AtlasRegConfigError(
            f'Parameter set "{desired_id}" for atlas "{atlas}" lacks a "registration" section.'
        )

    description = chosen.get('description', '')
    registration = _normalize_registration_dict(chosen['registration'])

    return AtlasRegParameterSet(
        atlas=atlas,
        parameter_id=desired_id,
        description=description,
        registration=registration,
        config_path=resolved_path,
    )


def _resolve_config_path(atlas: str, config_path: str | Path | None) -> Path:
    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.exists():
            raise AtlasRegConfigError(
                f'Custom atlas registration config "{resolved}" does not exist for atlas "{atlas}".'
            )
        return resolved

    try:
        pkg_path = Path(load_data(f'atlas_reg/{atlas}.yml'))
    except FileNotFoundError as exc:
        raise AtlasRegConfigError(
            f'No packaged atlas registration parameters found for atlas "{atlas}".'
        ) from exc
    return pkg_path


def _normalize_registration_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure structures are JSON-serializable and tuples converted to lists."""
    normalized: Dict[str, Any] = {}
    for key, value in config.items():
        normalized[key] = _serialize_for_json(value)
    return normalized


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = ('AtlasRegConfigError', 'AtlasRegParameterSet', 'load_parameter_set')
