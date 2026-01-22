# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Reportlets illustrating motion correction."""

from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import pandas as pd
from imageio import v2 as imageio
from nilearn import image
from nilearn.plotting import plot_epi
from nilearn.plotting.find_cuts import find_xyz_cut_coords
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class MotionPlotInputSpec(BaseInterfaceInputSpec):
    original_pet = File(
        exists=True,
        mandatory=True,
        desc='Original (uncorrected) PET series in native PET space',
    )
    corrected_pet = File(
        exists=True,
        mandatory=True,
        desc=(
            'Motion-corrected PET series derived by applying the estimated motion '
            'transforms to the original data in native PET space'
        ),
    )
    fd_file = File(exists=True, desc='Confounds file containing framewise displacement')
    duration = traits.Float(0.2, usedefault=True, desc='Frame duration for the GIF (seconds)')


class MotionPlotOutputSpec(TraitedSpec):
    svg_file = File(exists=True, desc='Animated before/after motion correction SVG')


class MotionPlot(SimpleInterface):
    """Generate animated visualizations before and after motion correction.

    A single GIF is created using ortho views with consistent cut coordinates
    and color scaling derived from the midpoint frame of each series. The
    per-frame views of the original and motion-corrected series are concatenated
    horizontally, allowing the main PET report to display the animation
    directly.
    """

    input_spec = MotionPlotInputSpec
    output_spec = MotionPlotOutputSpec

    def _run_interface(self, runtime):
        runtime.cwd = Path(runtime.cwd)

        svg_file = runtime.cwd / 'pet_motion_hmc.svg'
        svg_file.parent.mkdir(parents=True, exist_ok=True)

        mid_orig, cut_coords_orig, vmin_orig, vmax_orig = self._compute_display_params(
            self.inputs.original_pet
        )
        _, _, vmin_corr, vmax_corr = self._compute_display_params(self.inputs.corrected_pet)

        fd_values = None
        if isdefined(self.inputs.fd_file):
            fd_values = self._load_framewise_displacement(self.inputs.fd_file)

        svg_file = self._build_animation(
            output_path=svg_file,
            cut_coords_orig=cut_coords_orig,
            cut_coords_corr=cut_coords_orig,
            vmin_orig=vmin_orig,
            vmax_orig=vmax_orig,
            vmin_corr=vmin_corr,
            vmax_corr=vmax_corr,
            fd_values=fd_values,
        )

        self._results['svg_file'] = str(svg_file)

        return runtime

    def _compute_display_params(self, in_file: str):
        img = nib.load(in_file)
        if img.ndim == 3:
            mid_img = img
        else:
            mid_img = image.index_img(in_file, img.shape[-1] // 2)

        data = mid_img.get_fdata().astype(float)
        vmax = float(np.percentile(data.flatten(), 99.9))
        vmin = float(np.percentile(data.flatten(), 80))
        cut_coords = find_xyz_cut_coords(mid_img)

        return mid_img, cut_coords, vmin, vmax

    def _load_framewise_displacement(self, fd_file: str) -> np.ndarray:
        framewise_disp = pd.read_csv(fd_file, sep='\t')
        if 'framewise_displacement' in framewise_disp:
            fd_values = framewise_disp['framewise_displacement']
        elif 'FD' in framewise_disp:
            fd_values = framewise_disp['FD']
        else:
            available = ', '.join(framewise_disp.columns)
            raise ValueError(
                'Could not find framewise displacement column in confounds file '
                f'(available columns: {available})'
            )

        return np.asarray(fd_values.fillna(0.0), dtype=float)

    def _build_animation(
        self,
        *,
        output_path: Path,
        cut_coords_orig: tuple[float, float, float],
        cut_coords_corr: tuple[float, float, float],
        vmin_orig: float,
        vmax_orig: float,
        vmin_corr: float,
        vmax_corr: float,
        fd_values: np.ndarray | None,
    ) -> Path:
        orig_img = nib.load(self.inputs.original_pet)
        corr_img = nib.load(self.inputs.corrected_pet)

        n_frames = min(
            orig_img.shape[-1] if orig_img.ndim > 3 else 1,
            corr_img.shape[-1] if corr_img.ndim > 3 else 1,
        )

        if fd_values is not None:
            fd_values = np.asarray(fd_values[:n_frames], dtype=float)
            n_frames = min(n_frames, len(fd_values))

        with TemporaryDirectory() as tmpdir:
            frames = []
            for idx in range(n_frames):
                orig_png = Path(tmpdir) / f'orig_{idx:04d}.png'
                corr_png = Path(tmpdir) / f'corr_{idx:04d}.png'

                plot_epi(
                    image.index_img(self.inputs.original_pet, idx),
                    colorbar=True,
                    display_mode='ortho',
                    title=f'Before motion correction | Frame {idx + 1}',
                    cut_coords=cut_coords_orig,
                    vmin=vmin_orig,
                    vmax=vmax_orig,
                    output_file=str(orig_png),
                )
                plot_epi(
                    image.index_img(self.inputs.corrected_pet, idx),
                    colorbar=True,
                    display_mode='ortho',
                    title=f'After motion correction | Frame {idx + 1}',
                    cut_coords=cut_coords_corr,
                    vmin=vmin_corr,
                    vmax=vmax_corr,
                    output_file=str(corr_png),
                )

                orig_arr = np.asarray(imageio.imread(orig_png))
                corr_arr = np.asarray(imageio.imread(corr_png))

                max_height = max(orig_arr.shape[0], corr_arr.shape[0])
                if orig_arr.shape[0] < max_height:
                    pad = max_height - orig_arr.shape[0]
                    orig_arr = np.pad(
                        orig_arr, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=255
                    )
                if corr_arr.shape[0] < max_height:
                    pad = max_height - corr_arr.shape[0]
                    corr_arr = np.pad(
                        corr_arr, ((0, pad), (0, 0), (0, 0)), mode='constant', constant_values=255
                    )

                combined = np.concatenate([orig_arr, corr_arr], axis=1)
                frames.append(combined.astype(orig_arr.dtype, copy=False))

            width = int(frames[0].shape[1])
            frame_height = int(frames[0].shape[0])
            fd_height = 220 if fd_values is not None else 0
            height = frame_height + fd_height
            total_duration = self.inputs.duration * n_frames

            svg_parts = [
                '<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<style>',
                (
                    '.frame {'
                    f' opacity: 0; animation: framefade {total_duration}s infinite;'
                    ' animation-play-state: paused;'
                    '}'
                ),
                '.playing .frame {animation-play-state: running;}',
                '.fd-line-primary {fill: none; stroke: #2c7be5; stroke-width: 2;}',
                '.fd-line-alert {fill: none; stroke: #d7263d; stroke-width: 2;}',
                '.fd-axis {stroke: #333; stroke-width: 1;}',
                '.fd-point {fill: #2c7be5; stroke: white; stroke-width: 1;}',
                '#fd-marker {fill: #d7263d; stroke: white; stroke-width: 2;}',
                '#fd-value {font: 14px sans-serif; fill: #1a1a1a;}',
                '@keyframes framefade {0%, 80% {opacity: 1;} 100% {opacity: 0;}}',
            ]

            for idx in range(n_frames):
                delay = self.inputs.duration * idx
                svg_parts.append(f'.frame-{idx} {{animation-delay: {delay}s;}}')

            svg_parts.append('</style>')

            for idx, frame in enumerate(frames):
                buffer = BytesIO()
                imageio.imwrite(buffer, frame, format='PNG')
                data_uri = b64encode(buffer.getvalue()).decode('ascii')
                svg_parts.append(
                    f'<image class="frame frame-{idx}" '
                    f'width="{width}" height="{frame_height}" x="0" y="0" '
                    f'href="data:image/png;base64,{data_uri}" />'
                )

            if fd_values is not None:
                fd_padding = 45
                fd_chart_height = fd_height
                fd_x_start = fd_padding
                fd_x_end = width - fd_padding
                fd_axis_y = frame_height + fd_chart_height - fd_padding
                fd_axis_y_top = frame_height + fd_padding
                fd_y_range = fd_axis_y - fd_axis_y_top
                fd_max = float(np.nanmax(fd_values)) if np.any(fd_values) else 0.0
                if fd_max <= 0:
                    fd_max = 1.0

                x_scale = (fd_x_end - fd_x_start) / max(n_frames - 1, 1)
                points = []
                point_elems = []
                line_elems = []
                fd_threshold = 3.0
                for idx, value in enumerate(fd_values):
                    x_coord = fd_x_start + x_scale * idx
                    y_coord = fd_axis_y - (value / fd_max) * fd_y_range
                    points.append(f'{x_coord:.2f},{y_coord:.2f}')
                    point_elems.append(
                        f'<circle class="fd-point fd-point-{idx}" cx="{x_coord:.2f}" '
                        f'cy="{y_coord:.2f}" r="3" data-value="{value:.6f}" />'
                    )
                    if idx > 0:
                        prev_x, prev_y = map(float, points[idx - 1].split(','))
                        line_class = (
                            'fd-line-alert' if value >= fd_threshold else 'fd-line-primary'
                        )
                        line_elems.append(
                            f'<line class="{line_class}" x1="{prev_x:.2f}" y1="{prev_y:.2f}" '
                            f'x2="{x_coord:.2f}" y2="{y_coord:.2f}" />'
                        )

                fd_label_y = fd_axis_y_top + (fd_y_range / 2)
                fd_label_offset = 35

                tick_values = np.linspace(0, fd_max, num=3)
                tick_length = 6
                tick_elems = []
                label_elems = []
                for tick_value in tick_values:
                    y_coord = fd_axis_y - (tick_value / fd_max) * fd_y_range
                    tick_elems.append(
                        f'<line class="fd-axis" x1="{fd_x_start - tick_length}" '
                        f'x2="{fd_x_start}" y1="{y_coord:.2f}" y2="{y_coord:.2f}" />'
                    )
                    label_elems.append(
                        f'<text x="{fd_x_start - tick_length - 6}" y="{y_coord + 4:.2f}" '
                        'font-size="12" text-anchor="end">'
                        f'{tick_value:.1f}</text>'
                    )

                # X-axis ticks show every other frame (plus the last) to avoid clutter
                if n_frames <= 1:
                    x_tick_indices = np.array([0])
                else:
                    tick_stride = 2
                    x_tick_indices = np.arange(0, n_frames, tick_stride)
                    if x_tick_indices[-1] != n_frames - 1:
                        x_tick_indices = np.append(x_tick_indices, n_frames - 1)

                x_tick_length = 6
                x_tick_elems = []
                x_label_elems = []
                for tick_idx in x_tick_indices:
                    x_coord = fd_x_start + x_scale * tick_idx
                    x_tick_elems.append(
                        f'<line class="fd-axis" x1="{x_coord:.2f}" x2="{x_coord:.2f}" '
                        f'y1="{fd_axis_y}" y2="{fd_axis_y + x_tick_length}" />'
                    )
                    x_label_elems.append(
                        f'<text x="{x_coord:.2f}" y="{fd_axis_y + x_tick_length + 14}" '
                        'font-size="12" text-anchor="middle">'
                        f'{tick_idx + 1}</text>'
                    )

                svg_parts.extend(
                    [
                        '<g class="fd-plot" aria-label="Framewise displacement">',
                        f'<line class="fd-axis" x1="{fd_x_start}" x2="{fd_x_end}" '
                        f'y1="{fd_axis_y}" y2="{fd_axis_y}" />',
                        f'<line class="fd-axis" x1="{fd_x_start}" x2="{fd_x_start}" '
                        f'y1="{fd_axis_y_top}" y2="{fd_axis_y}" />',
                        *tick_elems,
                        *label_elems,
                        *line_elems,
                        *point_elems,
                        *x_tick_elems,
                        *x_label_elems,
                        f'<circle id="fd-marker" r="6" cx="{fd_x_start}" cy="{fd_axis_y}" />',
                        f'<text id="fd-value" x="{fd_x_start}" '
                        f'y="{fd_axis_y_top - 12}" aria-live="polite"></text>',
                        f'<text x="{fd_x_start - fd_label_offset}" y="{fd_label_y:.2f}" '
                        'font-size="14" text-anchor="middle" transform='
                        f'"rotate(-90 {fd_x_start - fd_label_offset},{fd_label_y:.2f})">'
                        'FD (mm)</text>',
                        f'<text x="{(fd_x_start + fd_x_end) / 2:.2f}" '
                        f'y="{fd_axis_y + 35}" font-size="14" text-anchor="middle">'
                        'Frames</text>',
                        '</g>',
                    ]
                )

            svg_parts.extend(
                [
                    '<script>',
                    '(() => {',
                    '  const svg = document.currentScript.parentNode;',
                    "  const frames = svg.querySelectorAll('.frame');",
                    "  const fdPoints = Array.from(svg.querySelectorAll('.fd-point'));",
                    "  const fdMarker = svg.querySelector('#fd-marker');",
                    "  const fdValueLabel = svg.querySelector('#fd-value');",
                    f'  const cycleMs = {total_duration * 1000:.0f};',
                    f'  const frameDurationMs = {self.inputs.duration * 1000:.0f};',
                    '  let restartTimer = null;',
                    '  let playbackTimer = null;',
                    '  let currentFrame = 0;',
                    '  const setFdMarker = (index) => {',
                    '    if (!fdMarker || !fdPoints.length) return;',
                    '    const point = fdPoints[index % fdPoints.length];',
                    '    fdMarker.setAttribute("cx", point.getAttribute("cx"));',
                    '    fdMarker.setAttribute("cy", point.getAttribute("cy"));',
                    '    if (fdValueLabel) {',
                    '      const value = parseFloat(point.dataset.value || "0");',
                    '      fdValueLabel.textContent = `Frame ${index + 1}: ${value.toFixed(3)} mm`;',
                    '    }',
                    '  };',
                    '  const showFrame = (index) => {',
                    '    currentFrame = index % frames.length;',
                    '    setFdMarker(currentFrame);',
                    '  };',
                    '  const restart = () => {',
                    '    frames.forEach((frame) => {',
                    "      frame.style.animation = 'none';",
                    '      // Force reflow to restart the CSS animation',
                    '      void frame.getBoundingClientRect();',
                    "      frame.style.animation = '';",
                    '    });',
                    '    showFrame(0);',
                    '  };',
                    '  const start = () => {',
                    '    if (restartTimer) {',
                    '      clearInterval(restartTimer);',
                    '    }',
                    '    if (playbackTimer) {',
                    '      clearInterval(playbackTimer);',
                    '    }',
                    '    svg.classList.add("playing");',
                    '    restart();',
                    '    restartTimer = setInterval(restart, cycleMs);',
                    '    playbackTimer = setInterval(() => {',
                    '      showFrame(currentFrame + 1);',
                    '    }, frameDurationMs);',
                    '  };',
                    '  const stop = () => {',
                    '    svg.classList.remove("playing");',
                    '    if (restartTimer) {',
                    '      clearInterval(restartTimer);',
                    '      restartTimer = null;',
                    '    }',
                    '    if (playbackTimer) {',
                    '      clearInterval(playbackTimer);',
                    '      playbackTimer = null;',
                    '    }',
                    '    frames.forEach((frame) => {',
                    "      frame.style.animation = 'none';",
                    '    });',
                    '  };',
                    '  showFrame(0);',
                    "  svg.addEventListener('mouseenter', start);",
                    "  svg.addEventListener('mouseleave', stop);",
                    '})();',
                    '</script>',
                    '</svg>',
                ]
            )

            output_path.write_text('\n'.join(svg_parts), encoding='utf-8')

        return output_path


__all__ = ['MotionPlot']
