from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nb  # noqa: E402
import numpy as np  # noqa: E402
from nibabel.processing import resample_from_to  # noqa: E402
from nibabel import as_closest_canonical  # noqa: E402
from nipype.interfaces.base import (  # noqa: E402
    BaseInterfaceInputSpec,
    Directory,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

SLICE_COUNT = 3
AXIS_ORDER = ('axial', 'coronal', 'sagittal')
AXIS_LABELS = {'axial': 'Axial', 'coronal': 'Coronal', 'sagittal': 'Sagittal'}

HTML_PLACEHOLDER = '<!-- ATLAS_REG_RUNS -->'


class _AtlasRegReportInputSpec(BaseInterfaceInputSpec):
    subject_id = traits.Str(mandatory=True, desc='Subject identifier (without ``sub-`` prefix)')
    atlas_name = traits.Str(mandatory=True, desc='Atlas identifier (e.g., `hammers`)')
    parameter_id = traits.Str(mandatory=True, desc='Registration parameter set identifier')
    parameter_description = traits.Str(desc='Human readable description of the parameter set')
    registration_params = traits.Dict(mandatory=True, desc='Registration parameters used in ANTs')
    config_path = File(exists=True, mandatory=True, desc='YAML configuration source for the parameters')
    runtime_seconds = traits.Float(
        desc='Elapsed wall-clock time for the registration stage (seconds)', mandatory=False
    )
    t1w_file = File(exists=True, mandatory=True, desc='T1w anatomical image in native space')
    template_file = File(exists=True, mandatory=True, desc='Atlas template image')
    atlas_file = File(exists=True, mandatory=True, desc='Atlas labels in template space')
    template_registered_file = File(
        exists=True, mandatory=True, desc='Atlas template warped into native (T1) space'
    )
    atlas_registered_file = File(
        exists=True, mandatory=True, desc='Atlas segmentation warped into native (T1) space'
    )
    subjects_dir = Directory(exists=True, mandatory=True, desc='FreeSurfer subjects directory')
    output_dir = Directory(exists=True, mandatory=True, desc='PETPrep derivatives root directory')
    run_uuid = traits.Str(mandatory=True, desc='Current PETPrep run UUID')
    custom_label = traits.Str(desc='Optional label appended to run identifiers')


class _AtlasRegReportOutputSpec(TraitedSpec):
    html_file = File(exists=True, desc='Atlas registration quality control HTML file')


class AtlasRegistrationReport(SimpleInterface):
    """Generate and append atlas registration visuals and metadata to an HTML QC report."""

    input_spec = _AtlasRegReportInputSpec
    output_spec = _AtlasRegReportOutputSpec

    def _run_interface(self, runtime):
        subject = self.inputs.subject_id
        output_root = Path(self.inputs.output_dir).resolve()
        if output_root.name != 'atlas_reg':
            atlas_root = output_root / 'atlas_reg'
        else:
            atlas_root = output_root
        atlas_root.mkdir(parents=True, exist_ok=True)

        subject_label = subject if subject.startswith('sub-') else f'sub-{subject}'
        subject_dir = atlas_root / subject_label
        figure_root = subject_dir / 'figures'
        figure_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        run_label = self.inputs.run_uuid
        if isdefined(self.inputs.custom_label) and self.inputs.custom_label:
            run_label = self.inputs.custom_label
        run_id = _sanitize_id(f'{self.inputs.atlas_name}_{self.inputs.parameter_id}_{run_label}_{timestamp}')
        run_dir = figure_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (
            t1_img,
            t1_data,
            template_img,
            template_data,
            atlas_data,
            template_reg_img,
            template_reg_data,
            atlas_reg_data,
            fs_seg_data,
        ) = self._load_data(subject)

        skull_sources = _find_skull_stripped_sources(subject_dir, subject_label)
        t1_brain_data: np.ndarray | None = None
        template_brain_data: np.ndarray | None = None
        t1_brain_range: tuple[float, float] | None = None
        template_brain_range: tuple[float, float] | None = None

        t1_brain_path = skull_sources.get('t1')
        if t1_brain_path is not None:
            try:
                t1_brain_data = _load_skull_stripped_data(t1_brain_path)
                t1_brain_range = _intensity_range(t1_brain_data)
            except Exception:
                t1_brain_data = None
                t1_brain_range = None

        template_brain_path = skull_sources.get('template')
        if template_brain_path is not None:
            try:
                template_brain_data = _load_skull_stripped_data(template_brain_path)
                template_brain_range = _intensity_range(template_brain_data)
            except Exception:
                template_brain_data = None
                template_brain_range = None

        t1_vmin, t1_vmax = _intensity_range(t1_data)
        template_vmin, template_vmax = _intensity_range(template_data)
        template_reg_vmin, template_reg_vmax = _intensity_range(template_reg_data)

        figure_map = _generate_figures(
            run_dir=run_dir,
            t1_data=t1_data,
            template_data=template_data,
            atlas_template_data=atlas_data,
            fs_seg_data=fs_seg_data,
            template_reg_data=template_reg_data,
            atlas_reg_data=atlas_reg_data,
            t1_range=(t1_vmin, t1_vmax),
            template_range=(template_vmin, template_vmax),
            template_reg_range=(template_reg_vmin, template_reg_vmax),
            t1_brain_data=t1_brain_data,
            template_brain_data=template_brain_data,
            t1_brain_range=t1_brain_range,
            template_brain_range=template_brain_range,
        )

        html_path = atlas_root / f'{subject_label}.html'
        if not html_path.exists():
            html_path.write_text(_build_html_header(subject), encoding='utf-8')

        relative_map = {
            key: Path(subject_label) / 'figures' / run_id / value.name
            for key, value in figure_map.items()
        }
        block = _render_block(
            atlas_name=self.inputs.atlas_name,
            parameter_id=self.inputs.parameter_id,
            parameter_description=self.inputs.parameter_description,
            config_path=self.inputs.config_path,
            runtime_seconds=self.inputs.runtime_seconds,
            registration_params=self.inputs.registration_params,
            relative_map=relative_map,
        )
        _append_block(html_path, block)
        self._results['html_file'] = str(html_path.resolve())
        return runtime

    def _load_data(self, subject: str):
        """Load and harmonize required images."""
        t1_img_orig = nb.load(self.inputs.t1w_file)
        t1_img = as_closest_canonical(t1_img_orig)
        t1_data = _ensure_3d(t1_img.get_fdata())

        template_img = as_closest_canonical(nb.load(self.inputs.template_file))
        template_data = _ensure_3d(template_img.get_fdata())

        atlas_img = as_closest_canonical(nb.load(self.inputs.atlas_file))
        atlas_template_data = np.rint(_ensure_3d(atlas_img.get_fdata())).astype(int)

        template_reg_img = as_closest_canonical(nb.load(self.inputs.template_registered_file))
        template_reg_data = _ensure_3d(template_reg_img.get_fdata())

        atlas_reg_img = as_closest_canonical(nb.load(self.inputs.atlas_registered_file))
        atlas_reg_data = np.rint(_ensure_3d(atlas_reg_img.get_fdata())).astype(int)

        subj_dir_name = subject if subject.startswith('sub-') else f'sub-{subject}'
        fs_seg_path = Path(self.inputs.subjects_dir) / subj_dir_name / 'mri' / 'aseg.mgz'
        if not fs_seg_path.exists():
            raise FileNotFoundError(
                f'FreeSurfer segmentation not found for subject sub-{subject}: {fs_seg_path}'
            )
        fs_seg_img = nb.load(str(fs_seg_path))
        fs_seg_img = as_closest_canonical(fs_seg_img)
        if (
            fs_seg_img.shape != t1_img.shape
            or not np.allclose(fs_seg_img.affine, t1_img.affine)
        ):
            fs_seg_img = resample_from_to(fs_seg_img, t1_img, order=0)
        fs_seg_data = np.rint(_ensure_3d(fs_seg_img.get_fdata())).astype(int)

        return (
            t1_img,
            t1_data,
            template_img,
            template_data,
            atlas_template_data,
            template_reg_img,
            template_reg_data,
            atlas_reg_data,
            fs_seg_data,
        )


def _ensure_3d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4:
        data = data[..., 0]
    return np.asarray(data)


def _intensity_range(data: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(data)
    if not np.any(finite):
        return float(np.nan), float(np.nan)
    return float(np.percentile(data[finite], 1)), float(np.percentile(data[finite], 99))


def _generate_figures(
    *,
    run_dir: Path,
    t1_data: np.ndarray,
    template_data: np.ndarray,
    atlas_template_data: np.ndarray,
    fs_seg_data: np.ndarray,
    template_reg_data: np.ndarray,
    atlas_reg_data: np.ndarray,
    t1_range: tuple[float, float],
    template_range: tuple[float, float],
    template_reg_range: tuple[float, float],
    t1_brain_data: np.ndarray | None = None,
    template_brain_data: np.ndarray | None = None,
    t1_brain_range: tuple[float, float] | None = None,
    template_brain_range: tuple[float, float] | None = None,
) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        't1_static': run_dir / 't1_static.png',
        'template_static': run_dir / 'template_static.png',
        'atlas_static': run_dir / 'atlas_static.png',
        't1_base': run_dir / 't1_base.png',
        'fs_overlay': run_dir / 'fs_overlay.png',
        'template_overlay': run_dir / 'template_overlay.png',
        'atlas_overlay': run_dir / 'atlas_overlay.png',
    }
    if t1_brain_data is not None:
        figure_paths['t1_brain'] = run_dir / 't1_brain.png'
    if template_brain_data is not None:
        figure_paths['template_brain'] = run_dir / 'template_brain.png'

    base_indices = _compute_slice_indices(t1_data.shape, SLICE_COUNT)
    template_indices = _compute_slice_indices(template_data.shape, SLICE_COUNT)
    atlas_template_indices = _compute_slice_indices(atlas_template_data.shape, SLICE_COUNT)

    _save_scalar_mosaic(t1_data, base_indices, figure_paths['t1_static'], vmin=t1_range[0], vmax=t1_range[1])
    _save_scalar_mosaic(
        template_data, template_indices, figure_paths['template_static'], vmin=template_range[0], vmax=template_range[1]
    )
    _save_segmentation_mosaic(atlas_template_data, atlas_template_indices, figure_paths['atlas_static'])

    # Base image reused for overlays
    _save_scalar_mosaic(t1_data, base_indices, figure_paths['t1_base'], vmin=t1_range[0], vmax=t1_range[1])
    _save_segmentation_overlay(fs_seg_data, base_indices, figure_paths['fs_overlay'])
    _save_scalar_overlay(
        template_reg_data, base_indices, figure_paths['template_overlay'], vmin=template_reg_range[0], vmax=template_reg_range[1]
    )
    _save_segmentation_overlay(atlas_reg_data, base_indices, figure_paths['atlas_overlay'])
    if t1_brain_data is not None:
        t1_brain_indices = _compute_slice_indices(t1_brain_data.shape, SLICE_COUNT)
        vmin, vmax = t1_brain_range if t1_brain_range is not None else (None, None)
        _save_scalar_mosaic(t1_brain_data, t1_brain_indices, figure_paths['t1_brain'], vmin=vmin, vmax=vmax)
    if template_brain_data is not None:
        template_brain_indices = _compute_slice_indices(template_brain_data.shape, SLICE_COUNT)
        vmin, vmax = template_brain_range if template_brain_range is not None else (None, None)
        _save_scalar_mosaic(
            template_brain_data,
            template_brain_indices,
            figure_paths['template_brain'],
            vmin=vmin,
            vmax=vmax,
        )
    return figure_paths


def _select_indices(size: int, count: int) -> list[int]:
    if size <= 1:
        return [0] * count
    if count == 1:
        return [size // 2]
    fractions = np.linspace(0.25, 0.75, count)
    indices: list[int] = []
    for frac in fractions:
        idx = int(round((size - 1) * frac))
        original_idx = idx
        while idx in indices and idx < size - 1:
            idx += 1
        if idx >= size:
            idx = original_idx
            while idx in indices and idx > 0:
                idx -= 1
        indices.append(max(0, min(size - 1, idx)))
    while len(indices) < count:
        indices.append(indices[-1])
    return indices[:count]


def _compute_slice_indices(shape: tuple[int, int, int], count: int) -> dict[str, list[int]]:
    return {
        'sagittal': _select_indices(shape[0], count),
        'coronal': _select_indices(shape[1], count),
        'axial': _select_indices(shape[2], count),
    }


def _extract_slices(data: np.ndarray, indices: dict[str, list[int]]) -> dict[str, list[np.ndarray]]:
    slices: dict[str, list[np.ndarray]] = {}
    for axis_name in AXIS_ORDER:
        axis_indices = indices[axis_name]
        axis_slices: list[np.ndarray] = []
        for idx in axis_indices:
            if axis_name == 'sagittal':
                slc = data[idx, :, :]
            elif axis_name == 'coronal':
                slc = data[:, idx, :]
            else:
                slc = data[:, :, idx]
            axis_slices.append(_reorient_slice(np.asarray(slc), axis_name))
        slices[axis_name] = axis_slices
    return slices


def _reorient_slice(slice_data: np.ndarray, axis_name: str) -> np.ndarray:
    view = np.array(slice_data)
    if axis_name == 'axial':
        view = np.rot90(view, k=1)
    elif axis_name == 'coronal':
        view = np.rot90(view, k=1)
    else:  # sagittal
        view = np.rot90(view, k=1)
    view = np.flipud(view)
    return view


def _save_scalar_mosaic(
    data: np.ndarray,
    indices: dict[str, list[int]],
    out_file: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    slices = _extract_slices(data, indices)
    rows = len(AXIS_ORDER)
    cols = len(indices[AXIS_ORDER[0]])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor='black')
    axes = np.atleast_2d(axes)
    for row, axis_name in enumerate(AXIS_ORDER):
        label = AXIS_LABELS[axis_name]
        for col, slc in enumerate(slices[axis_name]):
            ax = axes[row, col]
            ax.imshow(slc, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
            ax.axis('off')
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    ha='left',
                    va='top',
                    transform=ax.transAxes,
                )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def _save_segmentation_mosaic(
    data: np.ndarray, indices: dict[str, list[int]], out_file: Path
) -> None:
    slices = _extract_slices(data, indices)
    rows = len(AXIS_ORDER)
    cols = len(indices[AXIS_ORDER[0]])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor='black')
    axes = np.atleast_2d(axes)
    for row, axis_name in enumerate(AXIS_ORDER):
        label = AXIS_LABELS[axis_name]
        for col, slc in enumerate(slices[axis_name]):
            ax = axes[row, col]
            masked = np.ma.masked_where(slc <= 0, slc)
            ax.imshow(masked, cmap='tab20', origin='lower', interpolation='nearest')
            ax.axis('off')
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    ha='left',
                    va='top',
                    transform=ax.transAxes,
                )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def _save_segmentation_overlay(
    data: np.ndarray,
    indices: dict[str, list[int]],
    out_file: Path,
    alpha: float = 0.7,
) -> None:
    slices = _extract_slices(data, indices)
    rows = len(AXIS_ORDER)
    cols = len(indices[AXIS_ORDER[0]])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor='none')
    fig.patch.set_alpha(0)
    axes = np.atleast_2d(axes)
    for row, axis_name in enumerate(AXIS_ORDER):
        label = AXIS_LABELS[axis_name]
        for col, slc in enumerate(slices[axis_name]):
            ax = axes[row, col]
            masked = np.ma.masked_where(slc <= 0, slc)
            ax.imshow(masked, cmap='tab20', origin='lower', interpolation='nearest', alpha=alpha)
            ax.axis('off')
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    ha='left',
                    va='top',
                    transform=ax.transAxes,
                )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)



def _save_scalar_overlay(
    data: np.ndarray,
    indices: dict[str, list[int]],
    out_file: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.6,
) -> None:
    slices = _extract_slices(data, indices)
    rows = len(AXIS_ORDER)
    cols = len(indices[AXIS_ORDER[0]])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), facecolor='none')
    fig.patch.set_alpha(0)
    axes = np.atleast_2d(axes)
    for row, axis_name in enumerate(AXIS_ORDER):
        label = AXIS_LABELS[axis_name]
        for col, slc in enumerate(slices[axis_name]):
            ax = axes[row, col]
            ax.imshow(slc, cmap='gray', origin='lower', vmin=vmin, vmax=vmax, alpha=alpha)
            ax.axis('off')
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    label,
                    color='white',
                    fontsize=10,
                    fontweight='bold',
                    ha='left',
                    va='top',
                    transform=ax.transAxes,
                )
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(out_file, dpi=150, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def _find_skull_stripped_sources(subject_dir: Path, subject_label: str) -> dict[str, Path]:
    anat_dir = subject_dir / 'anat'
    if not anat_dir.exists():
        return {}

    def _pick(patterns: list[str]) -> Path | None:
        for pattern in patterns:
            matches = sorted(anat_dir.glob(pattern))
            for candidate in matches:
                if candidate.is_file():
                    return candidate
        return None

    t1_patterns = [
        f'{subject_label}_desc-brain_T1w.nii.gz',
        f'{subject_label}_desc-preproc_T1w.nii.gz',
    ]
    template_patterns = [
        f'{subject_label}_space-T1w_desc-*template*_T1w.nii.gz',
        f'{subject_label}_space-*_desc-brain_T1w.nii.gz',
        f'{subject_label}_space-*_desc-preproc_T1w.nii.gz',
    ]

    sources: dict[str, Path] = {}
    t1_path = _pick(t1_patterns)
    if t1_path is not None:
        sources['t1'] = t1_path
    template_path = _pick(template_patterns)
    if template_path is not None:
        sources['template'] = template_path
    return sources


def _load_skull_stripped_data(image_path: Path) -> np.ndarray:
    image = as_closest_canonical(nb.load(str(image_path)))
    data = _ensure_3d(image.get_fdata())

    skull_flag = _read_skull_stripped_flag(image_path)
    if skull_flag is True:
        return data

    mask_path = _infer_mask_path(image_path)
    if mask_path is None or not mask_path.exists():
        return data

    mask_img = as_closest_canonical(nb.load(str(mask_path)))
    if mask_img.shape != image.shape or not np.allclose(mask_img.affine, image.affine):
        mask_img = resample_from_to(mask_img, image, order=0)
    mask_data = _ensure_3d(mask_img.get_fdata()) > 0.5
    return np.where(mask_data, data, 0)


def _read_skull_stripped_flag(image_path: Path) -> bool | None:
    meta_path = _metadata_path(image_path)
    if meta_path is None or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    flag = metadata.get('SkullStripped')
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.strip().lower() == 'true'
    return None


def _metadata_path(image_path: Path) -> Path | None:
    if image_path.suffix == '.gz' and image_path.name.endswith('.nii.gz'):
        return image_path.with_suffix('').with_suffix('.json')
    return image_path.with_suffix('.json')


def _infer_mask_path(image_path: Path) -> Path | None:
    name = image_path.name
    parent = image_path.parent
    candidates: list[str] = []
    if '_desc-preproc_T1w' in name:
        candidates.append(name.replace('_desc-preproc_T1w', '_desc-brain_mask'))
    if '_desc-preproc_' in name and name.endswith('_T1w.nii.gz'):
        candidates.append(
            name.replace('_desc-preproc_', '_desc-brain_').replace('_T1w.nii.gz', '_mask.nii.gz')
        )
    if '_T1w' in name:
        candidates.append(name.replace('_T1w', '_mask'))
    for candidate in candidates:
        path = parent / candidate
        if path.exists():
            return path
    return None


def _build_html_header(subject: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Atlas registration QC – {subject}</title>
<style>
    body {{
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        background-color: #f5f5f5;
        margin: 2rem;
    }}
    h1 {{
        margin-bottom: 1.5rem;
    }}
    .run-block {{
        background-color: #ffffff;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        padding: 1.5rem;
        margin-bottom: 2rem;
    }}
    .grid {{
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
        margin-bottom: 1rem;
    }}
    .panel {{
        width: 100%;
    }}
    .panel h3 {{
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }}
    .panel img {{
        width: 100%;
        border-radius: 0.25rem;
        display: block;
    }}
    .overlay-container {{
        position: relative;
        width: 100%;
    }}
    .overlay-container img {{
        width: 100%;
        display: block;
        border-radius: 0.25rem;
    }}
    .overlay-container img.overlay {{
        position: absolute;
        top: 0;
        left: 0;
    }}
    .slider {{
        width: 100%;
        margin-top: 0.5rem;
    }}
    details {{
        margin-top: 0.75rem;
    }}
    pre {{
        background-color: #f0f0f0;
        padding: 0.75rem;
        border-radius: 0.25rem;
        overflow-x: auto;
    }}
</style>
<script>
function updateOverlay(id, value) {{
    const container = document.getElementById(id);
    if (!container) return;
    const overlay = container.querySelector('img.overlay');
    if (overlay) {{
        overlay.style.opacity = value;
    }}
}}
</script>
</head>
<body>
<h1>Atlas registration QC – {subject}</h1>
<!-- ATLAS_REG_RUNS -->
</body>
</html>
"""


def _render_block(
    *,
    atlas_name: str,
    parameter_id: str,
    parameter_description: str,
    config_path: str,
    runtime_seconds: float | None,
    registration_params: dict,
    relative_map: dict[str, Path],
) -> str:
    runtime_str = _format_runtime(runtime_seconds)
    params_json = json.dumps(registration_params, indent=2, sort_keys=True)
    slider_defaults = {
        'fs_overlay': 0.65,
        'template_overlay': 0.5,
        'atlas_overlay': 0.7,
    }

    def _slider_block(key: str, title: str) -> str:
        container_id = f'overlay-{relative_map["t1_base"].parent.stem}-{key}'
        base_src = relative_map['t1_base'].as_posix()
        overlay_src = relative_map[key].as_posix()
        default_opacity = slider_defaults.get(key, 0.6)
        return f"""
        <div class="panel">
            <h3>{title}</h3>
            <div class="overlay-container" id="{container_id}">
                <img src="{base_src}" alt="T1 reference" />
                <img src="{overlay_src}" class="overlay" alt="{title}" style="opacity:{default_opacity:.2f};" />
            </div>
            <input class="slider" type="range" min="0" max="1" step="0.05" value="{default_opacity:.2f}"
                oninput="updateOverlay('{container_id}', this.value)" />
        </div>
        """

    def _image_panel(title: str, key: str) -> str:
        if key not in relative_map:
            return ''
        return f"""
        <div class="panel">
            <h3>{title}</h3>
            <img src="{relative_map[key].as_posix()}" alt="{title}" />
        </div>
        """

    primary_panels: list[tuple[str, str]] = [
        ('Native T1', 't1_static'),
        ('Atlas template', 'template_static'),
        ('Atlas labels (template space)', 'atlas_static'),
    ]
    if 't1_brain' in relative_map:
        primary_panels.append(('Skull-stripped T1', 't1_brain'))
    if 'template_brain' in relative_map:
        primary_panels.append(('Skull-stripped template', 'template_brain'))
    primary_html = ''.join(_image_panel(title, key) for title, key in primary_panels if key in relative_map)

    return f"""
<div class="run-block">
    <h2>{atlas_name} – parameter set “{parameter_id}”</h2>
    <p><strong>Description:</strong> {parameter_description or 'n/a'}<br />
       <strong>Config:</strong> {Path(config_path).name} ({config_path})<br />
       <strong>Runtime:</strong> {runtime_str}</p>
    <div class="grid">
        {primary_html}
    </div>
    <div class="grid">
        {_slider_block('fs_overlay', 'FreeSurfer segmentation on T1')}
        {_slider_block('template_overlay', 'Template warped to T1')}
        {_slider_block('atlas_overlay', 'Atlas labels warped to T1')}
    </div>
    <details>
        <summary>Registration parameters</summary>
        <pre>{params_json}</pre>
    </details>
</div>
"""


def _append_block(html_path: Path, block: str) -> None:
    content = html_path.read_text(encoding='utf-8')
    if HTML_PLACEHOLDER not in content:
        # Fallback: append before closing body
        content = content.replace('</body>', block + '\n</body>')
    else:
        content = content.replace(HTML_PLACEHOLDER, block + '\n' + HTML_PLACEHOLDER, 1)
    html_path.write_text(content, encoding='utf-8')


def _format_runtime(runtime_seconds: float | None) -> str:
    if runtime_seconds is None or runtime_seconds <= 0:
        return 'n/a'
    minutes, seconds = divmod(int(round(runtime_seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:d}h {minutes:02d}m {seconds:02d}s'
    if minutes:
        return f'{minutes:d}m {seconds:02d}s'
    return f'{seconds:d}s'


def _sanitize_id(value: str) -> str:
    cleaned = ''.join(ch.lower() if ch.isalnum() else '-' for ch in value)
    cleaned = cleaned.strip('-')
    return cleaned or 'run'


__all__ = ('AtlasRegistrationReport',)
