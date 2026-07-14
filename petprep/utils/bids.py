# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Utilities to handle BIDS inputs."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path
from shutil import copytree, rmtree, which

import numpy as np
from bids.layout import BIDSLayout
from bids.utils import listify
from packaging.version import Version

from .. import config
from ..data import load as load_data


def get_sessions(layout: BIDSLayout, subject=None, **filters) -> list[str]:
    """Collect session labels, falling back to indexed file entities when needed.

    PyBIDS can return incorrect values from ``layout.get_sessions()`` when a dataset
    includes subject-level ``*_sessions.tsv`` files. Reading the ``session`` entity
    directly from indexed files avoids that collision.
    """

    sessions = None
    if hasattr(layout, 'get_sessions'):
        try:
            sessions = layout.get_sessions(subject=subject, **filters)
        except TypeError:
            sessions = layout.get_sessions(subject=subject)

    if sessions is not None:
        normalized = [
            session.removeprefix('ses-') for session in sessions if isinstance(session, str)
        ]
        if normalized:
            return sorted(normalized)
        if sessions and len(normalized) == len(sessions):
            return sorted(normalized)
        if not hasattr(layout, 'get'):
            return []

    entities = {'subject': subject, **filters}
    files = layout.get(
        return_type='object', **{k: v for k, v in entities.items() if v is not None}
    )
    sessions = {bids_file.entities.get('session') for bids_file in files}
    return sorted(session for session in sessions if session)


@cache
def _get_layout(derivatives_dir: Path) -> BIDSLayout:
    from petprep.data import load as load_data

    return BIDSLayout(derivatives_dir, config=[load_data('nipreps.json')], validate=False)


def collect_derivatives(
    derivatives_dir: Path,
    entities: dict,
    spec: dict | None = None,
    patterns: list[str] | None = None,
):
    """Gather existing derivatives and compose a cache."""
    if spec is None or patterns is None:
        _spec, _patterns = tuple(
            json.loads(load_data.readable('io_spec.json').read_text()).values()
        )

        if spec is None:
            spec = _spec
        if patterns is None:
            patterns = _patterns

    derivs_cache = defaultdict(list, {})
    layout = _get_layout(derivatives_dir)

    # search for both petrefs
    for k, q in spec['baseline'].items():
        query = {**entities, **q}
        item = _select_derivative_matches(
            layout.get(return_type='filename', **query), layout=layout
        )
        if not item:
            continue
        derivs_cache[f'{k}_petref'] = item[0] if len(item) == 1 else item
        # also store under generic key to simplify downstream checks
        if 'petref' not in derivs_cache:
            derivs_cache['petref'] = derivs_cache[f'{k}_petref']

    transforms_cache = {}
    for xfm, q in spec['transforms'].items():
        # Transform extension will often not match provided entities
        #   (e.g., ".nii.gz" vs ".txt").
        # And transform suffixes will be "xfm",
        #   whereas relevant src file will be "bold".
        query = {**entities, **q}
        item = _select_derivative_matches(
            layout.get(return_type='filename', **query), layout=layout
        )
        if not item:
            continue
        transforms_cache[xfm] = item[0] if len(item) == 1 else item
    derivs_cache['transforms'] = transforms_cache
    return derivs_cache


def _select_derivative_matches(candidates: list[str], *, layout: BIDSLayout):
    """Prefer the most appropriate derivative match for the current run context."""

    if len(candidates) < 2:
        return candidates

    combine_runs = getattr(config.workflow, 'combine_runs', False)
    if combine_runs:
        non_run = [path for path in candidates if 'run' not in layout.parse_file_entities(path)]
        if non_run:
            return [non_run[0]]

    return candidates


def write_bidsignore(deriv_dir):
    bids_ignore = (
        '*.html',
        'logs/',
        'figures/',  # Reports
        '*_xfm.*',  # Unspecified transform files
        '*.surf.gii',  # Unspecified structural outputs
        # Unspecified functional outputs
        '*_petref.nii.gz',
        '*_pet.pet.gii',
        '*_mixing.tsv',
        '*_timeseries.tsv',
        '*_tacs.tsv',
    )
    ignore_file = Path(deriv_dir) / '.bidsignore'

    ignore_file.write_text('\n'.join(bids_ignore) + '\n')


def write_derivative_description(bids_dir, deriv_dir, dataset_links=None):
    from .. import __version__

    DOWNLOAD_URL = f'https://github.com/nipreps/petprep/archive/{__version__}.tar.gz'

    bids_dir = Path(bids_dir)
    deriv_dir = Path(deriv_dir)
    desc = {
        'Name': 'PETPrep - PET PREProcessing workflow',
        'BIDSVersion': '1.4.0',
        'DatasetType': 'derivative',
        'GeneratedBy': [
            {
                'Name': 'PETPrep',
                'Version': __version__,
                'CodeURL': DOWNLOAD_URL,
            }
        ],
        'HowToAcknowledge': 'Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4), '
        'and include the generated citation boilerplate within the Methods '
        'section of the text.',
    }

    # Keys that can only be set by environment
    if 'PETPREP_DOCKER_TAG' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'docker',
            'Tag': f'nipreps/petprep:{os.environ["PETPREP_DOCKER_TAG"]}',
        }
    if 'PETPREP_SINGULARITY_URL' in os.environ:
        desc['GeneratedBy'][0]['Container'] = {
            'Type': 'singularity',
            'URI': os.getenv('PETPREP_SINGULARITY_URL'),
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = bids_dir / 'dataset_description.json'
    if fname.exists():
        orig_desc = json.loads(fname.read_text())

    if 'DatasetDOI' in orig_desc:
        desc['SourceDatasets'] = [
            {'URL': f'https://doi.org/{orig_desc["DatasetDOI"]}', 'DOI': orig_desc['DatasetDOI']}
        ]
    if 'License' in orig_desc:
        desc['License'] = orig_desc['License']

    # Add DatasetLinks
    if dataset_links:
        desc['DatasetLinks'] = {k: str(v) for k, v in dataset_links.items()}
        if 'templateflow' in dataset_links:
            desc['DatasetLinks']['templateflow'] = 'https://github.com/templateflow/templateflow'

    Path.write_text(deriv_dir / 'dataset_description.json', json.dumps(desc, indent=4))


def _ignore_run_pet_files(_, names):
    run_pet = []
    for name in names:
        if '_run-' not in name:
            continue
        if name.endswith('_pet.nii.gz') or name.endswith('_pet.nii') or name.endswith('_pet.json'):
            run_pet.append(name)
    return run_pet


_FRAMEWISE_METADATA = (
    'FrameReferenceTime',
    'ScaleFactor',
    'ScatterFraction',
    'DecayFactor',
    'DecayCorrectionFactor',
    'PromptRate',
    'SinglesRate',
    'RandomRate',
)
_DECAY_FACTOR_METADATA = ('DecayFactor', 'DecayCorrectionFactor')
_RADIONUCLIDE_HALF_LIVES = {
    'C11': 1220.4,
    'F18': 6586.2,
    'N13': 597.9,
    'O15': 122.24,
}
_RADIONUCLIDE_ALIASES = {
    'C11': ('C11', '11C', 'CARBON11', '11CARBON'),
    'F18': ('F18', '18F', 'FLUORINE18', '18FLUORINE'),
    'N13': ('N13', '13N', 'NITROGEN13', '13NITROGEN'),
    'O15': ('O15', '15O', 'OXYGEN15', '15OXYGEN'),
}
_SECONDS_PER_DAY = 24 * 3600
_TIMING_TOLERANCE_SECONDS = 1.0


def _parse_timezero(value) -> float | None:
    if not isinstance(value, str):
        return None

    match = re.match(r'^\s*((?:2[0-3]|[01]?\d)):([0-5]\d):([0-5]\d(?:\.\d+)?)\s*$', value)
    if not match:
        return None

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def _run_offset_from_injection_start(base_meta: dict, meta: dict) -> float | None:
    if 'InjectionStart' not in base_meta or 'InjectionStart' not in meta:
        return None
    try:
        offset = float(base_meta['InjectionStart']) - float(meta['InjectionStart'])
    except (TypeError, ValueError):
        return None
    return offset if np.isfinite(offset) and offset >= 0 else None


def _run_metadata_offset(base_meta: dict, meta: dict) -> float | None:
    """Return an exact run offset, validating independent timing fields when possible."""
    base_time = _parse_timezero(base_meta.get('TimeZero'))
    current_time = _parse_timezero(meta.get('TimeZero'))
    clock_offset = None
    if base_time is not None and current_time is not None:
        clock_offset = (current_time - base_time) % _SECONDS_PER_DAY

    injection_offset = _run_offset_from_injection_start(base_meta, meta)
    injection_fields_present = 'InjectionStart' in base_meta and 'InjectionStart' in meta
    if injection_fields_present and injection_offset is None:
        raise ValueError(
            'InjectionStart values must be numeric and place runs in chronological order'
        )

    if clock_offset is not None and injection_offset is not None:
        injection_clock_offset = injection_offset % _SECONDS_PER_DAY
        clock_difference = abs(clock_offset - injection_clock_offset)
        clock_difference = min(clock_difference, _SECONDS_PER_DAY - clock_difference)
        if clock_difference > _TIMING_TOLERANCE_SECONDS:
            raise ValueError(
                'TimeZero and InjectionStart imply inconsistent offsets; the runs may use '
                'different injections or incompatible timing references'
            )
        # InjectionStart retains elapsed-day information that a time-of-day value cannot encode.
        return injection_offset

    if injection_offset is not None:
        return injection_offset
    # TimeZero contains no date and cannot by itself distinguish same-day from multi-day runs.
    return None


def _frame_end(starts: list[float], durations: list[float]) -> float:
    if starts and durations and len(starts) == len(durations):
        return max(start + duration for start, duration in zip(starts, durations, strict=True))
    if starts:
        return max(starts) + (float(sum(durations)) if durations else 0.0)
    return 0.0


def _run_time_offsets_with_reliability(metas: list[dict]) -> tuple[list[float], list[bool]]:
    offsets = []
    reliable = []
    fallback_offset = 0.0
    base_meta = metas[0] if metas else {}

    for run_index, meta in enumerate(metas):
        starts = [float(start) for start in (meta.get('FrameTimesStart') or [])]
        durations = [float(duration) for duration in (meta.get('FrameDuration') or [])]
        starts_are_relative = bool(starts) and np.isclose(min(starts), 0.0)

        try:
            metadata_offset = _run_metadata_offset(base_meta, meta)
        except ValueError as exc:
            raise ValueError(f'Cannot safely combine PET run {run_index + 1}: {exc}') from exc

        if metadata_offset is not None:
            offset = metadata_offset
            adjusted_starts = [start + offset for start in starts]
            is_reliable = True
        elif starts_are_relative or not starts:
            offset = fallback_offset
            adjusted_starts = [start + offset for start in starts]
            is_reliable = run_index == 0
        else:
            offset = 0.0
            adjusted_starts = starts
            is_reliable = run_index == 0

        if (
            run_index > 0
            and is_reliable
            and adjusted_starts
            and min(adjusted_starts) < fallback_offset - _TIMING_TOLERANCE_SECONDS
        ):
            raise ValueError(
                f'Cannot safely combine PET run {run_index + 1}: adjusted frame times overlap '
                'or precede the prior run'
            )

        offsets.append(float(offset))
        reliable.append(is_reliable)
        frame_end = _frame_end(adjusted_starts, durations)
        if not adjusted_starts and durations:
            frame_end = offset + float(sum(durations))
        fallback_offset = max(fallback_offset, frame_end)

    return offsets, reliable


def _run_time_offsets(metas: list[dict]) -> list[float]:
    return _run_time_offsets_with_reliability(metas)[0]


def _require_reliable_run_timing(reliable: list[bool]) -> None:
    unresolved_runs = [str(index + 1) for index, value in enumerate(reliable) if not value]
    if unresolved_runs:
        raise ValueError(
            'Cannot safely combine PET runs because an exact timing offset could not be '
            f'determined for run(s) {", ".join(unresolved_runs)}. Provide consistent TimeZero '
            'and/or InjectionStart metadata; assuming contiguous runs could erase acquisition '
            'gaps and produce incorrect decay correction.'
        )


def _metadata_as_framewise(value, frame_count: int) -> list | None:
    if frame_count == 0 or not isinstance(value, list):
        return None
    if len(value) == frame_count:
        return value
    if len(value) == 1:
        return value * frame_count
    return None


def _infer_radionuclide(value) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = re.sub(r'[^A-Za-z0-9]', '', value).upper()
    matches = {
        radionuclide
        for radionuclide, aliases in _RADIONUCLIDE_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    }
    return matches.pop() if len(matches) == 1 else None


def _metadata_half_life(meta: dict) -> float | None:
    try:
        half_life = float(meta['RadionuclideHalfLife'])
    except (KeyError, TypeError, ValueError):
        half_life = None
    radionuclide = _infer_radionuclide(meta.get('TracerRadionuclide'))
    inferred_half_life = _RADIONUCLIDE_HALF_LIVES.get(radionuclide)
    if half_life is not None and np.isfinite(half_life) and half_life > 0:
        if inferred_half_life is not None and not np.isclose(
            half_life, inferred_half_life, rtol=0.01
        ):
            return None
        return half_life
    return inferred_half_life


def _absolute_decay_correction_times(
    metas: list[dict], run_offsets: list[float]
) -> list[float] | None:
    if not metas or not all(meta.get('ImageDecayCorrected') is True for meta in metas):
        return None

    decay_times = []
    for meta, run_offset in zip(metas, run_offsets, strict=True):
        try:
            decay_time = float(meta['ImageDecayCorrectionTime']) + run_offset
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(decay_time):
            return None
        decay_times.append(decay_time)
    return decay_times


def _decay_rescale_factors(metas: list[dict], run_offsets: list[float]) -> list[float]:
    factors = [1.0] * len(metas)
    if not metas:
        return factors

    corrected_values = [meta.get('ImageDecayCorrected') for meta in metas]
    if not all(isinstance(value, bool) for value in corrected_values):
        raise ValueError(
            'Cannot safely combine PET runs unless ImageDecayCorrected is defined as a boolean '
            'for every run'
        )
    correction_times_are_valid = True
    for meta in metas:
        try:
            correction_time = float(meta['ImageDecayCorrectionTime'])
        except (KeyError, TypeError, ValueError):
            correction_times_are_valid = False
            break
        if not np.isfinite(correction_time):
            correction_times_are_valid = False
            break
    if not correction_times_are_valid:
        raise ValueError(
            'Cannot safely combine PET runs unless ImageDecayCorrectionTime is defined as a '
            'finite number for every run'
        )
    if all(value is False for value in corrected_values):
        return factors
    if not all(value is True for value in corrected_values):
        raise ValueError('Cannot safely combine decay-corrected and uncorrected PET runs')

    decay_times = _absolute_decay_correction_times(metas, run_offsets)
    if decay_times is None:
        raise ValueError(
            'Cannot safely combine decay-corrected PET runs unless '
            'ImageDecayCorrectionTime is defined for every run'
        )

    if np.allclose(
        decay_times,
        decay_times[0],
        rtol=0.0,
        atol=_TIMING_TOLERANCE_SECONDS,
    ):
        return factors

    half_lives = [_metadata_half_life(meta) for meta in metas]
    if any(half_life is None for half_life in half_lives):
        raise ValueError(
            'Cannot safely rescale decay-corrected PET runs with different correction times '
            'because the radionuclide half-life is missing, unsupported, or inconsistent with '
            'TracerRadionuclide'
        )
    half_life = _metadata_half_life(metas[0])
    half_lives_match = all(
        np.isclose(half_life, other_half_life, rtol=0.01) for other_half_life in half_lives
    )
    if not half_lives_match:
        raise ValueError(
            'Cannot safely rescale decay-corrected PET runs with inconsistent radionuclide '
            'half-lives'
        )
    decay_constant = np.log(2.0) / half_life
    target_decay_time = decay_times[0]
    for i, decay_time in enumerate(decay_times):
        with np.errstate(over='ignore', under='ignore'):
            factor = float(np.exp(decay_constant * (decay_time - target_decay_time)))
        if not np.isfinite(factor) or factor == 0.0:
            raise ValueError('Decay rescaling factor is outside the supported numeric range')
        factors[i] = factor

    return factors


def _merge_decay_correction_metadata(merged: dict, metas: list[dict]) -> None:
    corrected_values = [meta.get('ImageDecayCorrected') for meta in metas]
    if all(value is False for value in corrected_values):
        merged['ImageDecayCorrected'] = False
        merged['ImageDecayCorrectionTime'] = float(metas[0]['ImageDecayCorrectionTime'])
        return

    merged['ImageDecayCorrected'] = True
    merged['ImageDecayCorrectionTime'] = float(metas[0]['ImageDecayCorrectionTime'])


def _merge_offset_timing_metadata(
    metas: list[dict], run_offsets: list[float], key: str
) -> list[float] | None:
    values = []
    for meta, run_offset in zip(metas, run_offsets, strict=True):
        starts = meta.get(key) or []
        if not starts:
            return None
        values.extend([float(start) + run_offset for start in starts])
    return values


def _merge_frame_metadata(
    metas: list[dict],
    *,
    run_offsets: list[float] | None = None,
    decay_rescale_factors: list[float] | None = None,
) -> dict:
    merged = metas[0].copy()
    frame_durations = []
    run_offsets = run_offsets or _run_time_offsets(metas)
    expected_decay_rescale_factors = _decay_rescale_factors(metas, run_offsets)
    if decay_rescale_factors is None:
        decay_rescale_factors = expected_decay_rescale_factors
    elif not np.allclose(decay_rescale_factors, expected_decay_rescale_factors):
        raise ValueError('Decay rescale factors do not match the PET decay metadata')

    frame_times = _merge_offset_timing_metadata(metas, run_offsets, 'FrameTimesStart')
    volume_timing = _merge_offset_timing_metadata(metas, run_offsets, 'VolumeTiming')

    for meta in metas:
        durations = meta.get('FrameDuration') or []

        if durations:
            frame_durations.extend(durations)

    if frame_times:
        merged['FrameTimesStart'] = frame_times
    else:
        merged.pop('FrameTimesStart', None)
    if volume_timing:
        merged['VolumeTiming'] = volume_timing
    else:
        merged.pop('VolumeTiming', None)
    if frame_durations:
        merged['FrameDuration'] = frame_durations
        merged['AcquisitionDuration'] = float(sum(frame_durations))

    _merge_decay_correction_metadata(merged, metas)

    for key in _FRAMEWISE_METADATA:
        values = []
        for meta, run_offset, decay_factor in zip(
            metas, run_offsets, decay_rescale_factors, strict=True
        ):
            frame_count = max(
                len(meta.get('FrameTimesStart') or []),
                len(meta.get('FrameDuration') or []),
            )
            framewise_values = _metadata_as_framewise(meta.get(key), frame_count)
            if framewise_values is None:
                values = []
                break

            if key == 'FrameReferenceTime':
                values.extend([float(value) + run_offset for value in framewise_values])
            elif key in _DECAY_FACTOR_METADATA:
                values.extend([float(value) * decay_factor for value in framewise_values])
            else:
                values.extend(framewise_values)

        if values:
            merged[key] = values
        else:
            merged.pop(key, None)

    return merged


def combine_pet_runs(bids_dir: Path, layout: BIDSLayout, work_dir: Path, subjects, bids_filters):
    import nibabel as nb
    from nipype.interfaces.freesurfer.model import Concatenate

    combined_root = Path(work_dir) / 'combined_bids'
    if combined_root.exists():
        rmtree(combined_root)
    combined_root.mkdir(exist_ok=True, parents=True)

    copytree(
        bids_dir, combined_root, symlinks=True, dirs_exist_ok=True, ignore=_ignore_run_pet_files
    )

    pet_filters = (bids_filters or {}).get('pet', {})
    pet_filters = {key: value for key, value in pet_filters.items() if key != 'run'}

    combined_files = []

    for subject in subjects:
        pet_files = layout.get(
            subject=subject,
            suffix='pet',
            extension=['.nii', '.nii.gz'],
            return_type='filename',
            **pet_filters,
        )

        if not pet_files:
            continue

        grouped: defaultdict[tuple, list[str]] = defaultdict(list)
        for pet_file in pet_files:
            entities = layout.parse_file_entities(pet_file)
            entities.pop('run', None)
            entities.pop('suffix', None)
            entities.pop('extension', None)
            entities.pop('datatype', None)
            entities.pop('space', None)
            key = tuple(sorted(entities.items()))
            grouped[key].append(pet_file)

        for files in grouped.values():
            files = sorted(
                files,
                key=lambda path: (
                    layout.parse_file_entities(path).get('run')
                    or layout.parse_file_entities(path).get('acq')
                    or path
                ),
            )
            imgs = [nb.load(file) for file in files]
            metas = [layout.get_metadata(file) for file in files]
            run_offsets, reliable_timing = _run_time_offsets_with_reliability(metas)
            _require_reliable_run_timing(reliable_timing)
            decay_rescale_factors = _decay_rescale_factors(metas, run_offsets)

            if imgs:
                shapes = [img.shape for img in imgs]

                if any(len(shape) < 3 or len(shape) > 4 for shape in shapes):
                    raise ValueError('PET images must be 3D or 4D when combining runs')

                spatial_shape = shapes[0][:3]
                if any(shape[:3] != spatial_shape for shape in shapes):
                    raise ValueError(
                        'PET images must match in spatial dimensions when combining runs'
                    )

            original = Path(files[0])
            rel_path = original.relative_to(bids_dir)
            new_name = re.sub(r'_run-[^_]+', '', rel_path.name)
            output_img = combined_root / rel_path.with_name(new_name)
            output_img.parent.mkdir(exist_ok=True, parents=True)
            needs_rescaling = not np.allclose(decay_rescale_factors, 1.0)
            if which('mri_concat') and not needs_rescaling:
                concat = Concatenate(in_files=files, concatenated_file=str(output_img))
                concat.run()
            else:
                normalized_imgs = []
                for img, rescale_factor in zip(imgs, decay_rescale_factors, strict=True):
                    if img.ndim == 3:
                        data = np.expand_dims(img.get_fdata(dtype=np.float32), axis=3)
                        if not np.isclose(rescale_factor, 1.0):
                            data = data * np.float32(rescale_factor)
                        header = img.header.copy()
                        header.set_data_shape(data.shape)
                        header.set_data_dtype(np.float32)
                        normalized_imgs.append(nb.Nifti1Image(data, img.affine, header))
                    elif not np.isclose(rescale_factor, 1.0):
                        data = img.get_fdata(dtype=np.float32) * np.float32(rescale_factor)
                        header = img.header.copy()
                        header.set_data_dtype(np.float32)
                        normalized_imgs.append(nb.Nifti1Image(data, img.affine, header))
                    else:
                        normalized_imgs.append(img)
                combined_img = nb.concat_images(normalized_imgs, axis=3)
                nb.save(combined_img, str(output_img))

            combined_meta = _merge_frame_metadata(
                metas,
                run_offsets=run_offsets,
                decay_rescale_factors=decay_rescale_factors,
            )
            meta_output = output_img.with_suffix('').with_suffix('.json')
            meta_output.write_text(json.dumps(combined_meta, indent=4))
            combined_files.append(str(output_img))

    return combined_root, combined_files


def get_subject_modality_status(
    bids_dir: Path,
    subject_id: str,
    *,
    bids_filters: dict | None = None,
    derivatives: dict | None = None,
    anat_only: bool = False,
) -> dict[str, bool]:
    """Return subject-level PET/T1w availability after applying active filters."""
    from niworkflows.utils.bids import DEFAULT_BIDS_QUERIES, collect_data

    queries = copy.deepcopy(DEFAULT_BIDS_QUERIES)
    queries['t1w'].pop('datatype', None)
    subject_data = collect_data(
        bids_dir,
        subject_id,
        bids_filters=bids_filters,
        queries=queries,
    )[0]

    has_pet = anat_only or bool(subject_data['pet'])
    has_t1w = bool(subject_data['t1w'])

    if not has_t1w and derivatives:
        from smriprep.utils.bids import collect_derivatives as collect_anat_derivatives

        anatomical_cache = {}
        for deriv_dir in derivatives.values():
            anatomical_cache.update(
                collect_anat_derivatives(
                    derivatives_dir=deriv_dir,
                    subject_id=subject_id,
                    std_spaces=[],
                )
            )
        has_t1w = 't1w_preproc' in anatomical_cache

    return {'pet': has_pet, 't1w': has_t1w}


def validate_input_dir(exec_env, bids_dir, participant_label, need_T1w=True):
    # Ignore issues and warnings that should not influence PETPrep
    import subprocess
    import tempfile

    validator_config_dict = {
        'ignore': [
            'EVENTS_COLUMN_ONSET',
            'EVENTS_COLUMN_DURATION',
            'TSV_EQUAL_ROWS',
            'TSV_EMPTY_CELL',
            'TSV_IMPROPER_NA',
            'VOLUME_COUNT_MISMATCH',
            'BVAL_MULTIPLE_ROWS',
            'BVEC_NUMBER_ROWS',
            'DWI_MISSING_BVAL',
            'INCONSISTENT_SUBJECTS',
            'INCONSISTENT_PARAMETERS',
            'BVEC_ROW_LENGTH',
            'B_FILE',
            'PARTICIPANT_ID_COLUMN',
            'PARTICIPANT_ID_MISMATCH',
            'TASK_NAME_MUST_DEFINE',
            'PHENOTYPE_SUBJECTS_MISSING',
            'STIMULUS_FILE_MISSING',
            'DWI_MISSING_BVEC',
            'EVENTS_TSV_MISSING',
            'TSV_IMPROPER_NA',
            'ACQTIME_FMT',
            'Participants age 89 or higher',
            'DATASET_DESCRIPTION_JSON_MISSING',
            'FILENAME_COLUMN',
            'WRONG_NEW_LINE',
            'MISSING_TSV_COLUMN_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_CHANNELS',
            'MISSING_TSV_COLUMN_IEEG_ELECTRODES',
            'UNUSED_STIMULUS',
            'CHANNELS_COLUMN_SFREQ',
            'CHANNELS_COLUMN_LOWCUT',
            'CHANNELS_COLUMN_HIGHCUT',
            'CHANNELS_COLUMN_NOTCH',
            'CUSTOM_COLUMN_WITHOUT_DESCRIPTION',
            'ACQTIME_FMT',
            'SUSPICIOUSLY_LONG_EVENT_DESIGN',
            'SUSPICIOUSLY_SHORT_EVENT_DESIGN',
            'MALFORMED_BVEC',
            'MALFORMED_BVAL',
            'MISSING_TSV_COLUMN_EEG_ELECTRODES',
            'MISSING_SESSION',
        ],
        'error': ['NO_T1W'] if need_T1w else [],
        'ignoredFiles': ['/dataset_description.json', '/participants.tsv'],
    }
    # Limit validation only to data from requested participants
    if participant_label:
        all_subs = {s.name[4:] for s in bids_dir.glob('sub-*')}
        selected_subs = {s.removeprefix('sub-') for s in participant_label}
        bad_labels = selected_subs.difference(all_subs)
        if bad_labels:
            error_msg = (
                'Data for requested participant(s) label(s) not found. Could '
                'not find data for participant(s): %s. Please verify the requested '
                'participant labels.'
            )
            if exec_env == 'docker':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the docker container. Please make sure all '
                    'volumes are mounted properly (see https://docs.docker.com/'
                    'engine/reference/commandline/run/#mount-volume--v---read-only)'
                )
            if exec_env == 'singularity':
                error_msg += (
                    ' This error can be caused by the input data not being '
                    'accessible inside the singularity container. Please make sure '
                    'all paths are mapped properly (see https://www.sylabs.io/'
                    'guides/3.0/user-guide/bind_paths_and_mounts.html)'
                )
            raise RuntimeError(error_msg % ','.join(bad_labels))

        ignored_subs = all_subs.difference(selected_subs)
        if ignored_subs:
            for sub in ignored_subs:
                validator_config_dict['ignoredFiles'].append(f'/sub-{sub}/**')
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp:
        temp.write(json.dumps(validator_config_dict))
        temp.flush()
        try:
            subprocess.check_call(['bids-validator', str(bids_dir), '-c', temp.name])  # noqa: S607
        except FileNotFoundError:
            print('bids-validator does not appear to be installed', file=sys.stderr)


def check_pipeline_version(pipeline_name, cvers, data_desc):
    """
    Search for existing BIDS pipeline output and compares against current pipeline version.

    .. testsetup::

        >>> import json
        >>> data = {"GeneratedBy": [{"Name": "PETPrep", "Version": "0.0.5"}]}
        >>> desc_file = Path('sample_dataset_description.json')
        >>> _ = desc_file.write_text(json.dumps(data))

        >>> data = {"PipelineDescription": {"Version": "1.1.1rc5"}}
        >>> desc_file = Path('legacy_dataset_description.json')
        >>> _ = desc_file.write_text(json.dumps(data))

    Parameters
    ----------
    cvers : :obj:`str`
        Current pipeline version
    data_desc : :obj:`str` or :obj:`os.PathLike`
        Path to pipeline output's ``dataset_description.json``

    Examples
    --------
    >>> check_pipeline_version('PETPrep', '0.0.5', 'sample_dataset_description.json')
    >>> check_pipeline_version(
    ...     'PETPrep', '0.0.5+gb2e14d98', 'sample_dataset_description.json'
    ... )
    >>> check_pipeline_version('PETPrep', '24.0.0', 'sample_dataset_description.json')
    'Previous output generated by version 0.0.5 found.'
    >>> check_pipeline_version(
    ...     'PETPrep', '24.0.0', 'legacy_dataset_description.json'
    ... )  # doctest: +ELLIPSIS
    'Previous output generated by version 1.1.1rc5 found.'

    Returns
    -------
    message : :obj:`str` or :obj:`None`
        A warning string if there is a difference between versions, otherwise ``None``.

    """
    data_desc = Path(data_desc)
    if not data_desc.exists():
        return

    desc = json.loads(data_desc.read_text())
    generators = {
        generator['Name']: generator.get('Version', '0+unknown')
        for generator in desc.get('GeneratedBy', [])
    }
    dvers = generators.get(pipeline_name)
    if dvers is None:
        # Very old style
        dvers = desc.get('PipelineDescription', {}).get('Version', '0+unknown')
    if Version(cvers).public != Version(dvers).public:
        return f'Previous output generated by version {dvers} found.'


def extract_entities(file_list):
    """
    Return a dictionary of common entities given a list of files.

    Examples
    --------
    >>> extract_entities("sub-01/anat/sub-01_T1w.nii.gz")
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_T1w.nii.gz"] * 2)
    {'subject': '01', 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}
    >>> extract_entities(["sub-01/anat/sub-01_run-1_T1w.nii.gz",
    ...                   "sub-01/anat/sub-01_run-2_T1w.nii.gz"])
    {'subject': '01', 'run': [1, 2], 'suffix': 'T1w', 'datatype': 'anat', 'extension': '.nii.gz'}

    """
    from collections import defaultdict

    from bids.layout import parse_file_entities

    entities = defaultdict(list)
    for e, v in [
        ev_pair for f in listify(file_list) for ev_pair in parse_file_entities(f).items()
    ]:
        entities[e].append(v)

    def _unique(inlist):
        inlist = sorted(set(inlist))
        if len(inlist) == 1:
            return inlist[0]
        return inlist

    return {k: _unique(v) for k, v in entities.items()}


def _find_nearest_path(path_dict, input_path):
    """Find the nearest relative path from an input path to a dictionary of paths.

    If ``input_path`` is not relative to any of the paths in ``path_dict``,
    the absolute path string is returned.

    If ``input_path`` is already a BIDS-URI, then it will be returned unmodified.

    Parameters
    ----------
    path_dict : dict of (str, Path)
        A dictionary of paths.
    input_path : Path
        The input path to match.

    Returns
    -------
    matching_path : str
        The nearest relative path from the input path to a path in the dictionary.
        This is either the concatenation of the associated key from ``path_dict``
        and the relative path from the associated value from ``path_dict`` to ``input_path``,
        or the absolute path to ``input_path`` if no matching path is found from ``path_dict``.

    Examples
    --------
    >>> from pathlib import Path
    >>> path_dict = {
    ...     'bids::': Path('/data/derivatives/petprep'),
    ...     'bids:raw:': Path('/data'),
    ...     'bids:deriv-0:': Path('/data/derivatives/source-1'),
    ... }
    >>> input_path = Path('/data/derivatives/source-1/sub-01/pet/sub-01_pet.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # match to 'bids:deriv-0:'
    'bids:deriv-0:sub-01/pet/sub-01_pet.nii.gz'
    >>> input_path = Path('/out/sub-01/pet/sub-01_pet.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # no match- absolute path
    '/out/sub-01/pet/sub-01_pet.nii.gz'
    >>> input_path = Path('/data/sub-01/pet/sub-01_pet.nii.gz')
    >>> _find_nearest_path(path_dict, input_path)  # match to 'bids:raw:'
    'bids:raw:sub-01/pet/sub-01_pet.nii.gz'
    >>> input_path = 'bids::sub-01/pet/sub-01_pet.nii.gz'
    >>> _find_nearest_path(path_dict, input_path)  # already a BIDS-URI
    'bids::sub-01/pet/sub-01_pet.nii.gz'
    """
    # Don't modify BIDS-URIs
    if isinstance(input_path, str) and input_path.startswith('bids:'):
        return input_path

    input_path = Path(input_path)
    matching_path = None
    for key, path in path_dict.items():
        if input_path.is_relative_to(path):
            relative_path = input_path.relative_to(path)
            if (matching_path is None) or (len(relative_path.parts) < len(matching_path.parts)):
                matching_key = key
                matching_path = relative_path

    if matching_path is None:
        matching_path = str(input_path.absolute())
    else:
        matching_path = f'{matching_key}{matching_path}'

    return matching_path
