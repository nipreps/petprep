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
"""Interfaces to generate reportlets."""

import io
import logging
import os
import re
import time
from collections import Counter
from uuid import uuid4

import nibabel as nb
import numpy as np
import svgutils.transform as svgt
from nilearn import image as nlimage
from nilearn.plotting import plot_anat

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    Directory,
    File,
    InputMultiObject,
    SimpleInterface,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)
from nireports.reportlets.utils import compose_view, cuts_from_bbox, extract_svg, robust_set_limits
from nireports.tools.ndimage import rotate_affine, rotation2canonical
from smriprep.interfaces.freesurfer import ReconAll

LOGGER = logging.getLogger('nipype.interface')

_ORI_TO_NAME = {
    'L': 'Left',
    'R': 'Right',
    'A': 'Anterior',
    'P': 'Posterior',
    'S': 'Superior',
    'I': 'Inferior',
}

_OPPOSITE = {
    'L': 'R',
    'R': 'L',
    'A': 'P',
    'P': 'A',
    'S': 'I',
    'I': 'S',
}

def _plot_registration_with_overlays(
    anat_nii,
    div_id,
    plot_params=None,
    order=('z', 'x', 'y'),
    cuts=None,
    estimate_brightness=False,
    label=None,
    contour=None,
    compress='auto',
    dismiss_affine=False,
    overlays=None,
):
    """Local copy of NiReports plot_registration that handles the atlas mask overlay."""

    plot_params = {} if plot_params is None else dict(plot_params)
    if cuts is None:
        raise ValueError('Slice locations are required to plot the registration.')

    anat_nii = nb.Nifti1Image.from_image(anat_nii)

    overlay_images = []
    if overlays:
        overlay = overlays[0]
        params = dict(overlay.get('params', {}))
        overlay_img = nb.Nifti1Image.from_image(overlay['image'])
        overlay_images.append((overlay_img, params))

    if estimate_brightness:
        plot_params = robust_set_limits(np.asanyarray(anat_nii.dataobj).reshape(-1), plot_params)

    ribbon = False
    if contour is not None:
        contour = nb.Nifti1Image.from_image(contour)
        ribbon = np.array_equal(np.unique(contour.get_fdata()), [0, 2, 3, 41, 42])
        if ribbon:
            contour_data = contour.get_fdata() % 39
            white = nlimage.new_img_like(contour, contour_data == 2)
            pial = nlimage.new_img_like(contour, contour_data >= 2)

    if dismiss_affine:
        canonical_r = rotation2canonical(anat_nii)
        anat_nii = rotate_affine(anat_nii, rot=canonical_r)
        if contour is not None:
            contour = rotate_affine(contour, rot=canonical_r)
        if ribbon:
            white = rotate_affine(white, rot=canonical_r)
            pial = rotate_affine(pial, rot=canonical_r)
        rotated_overlays = []
        for overlay_img, params in overlay_images:
            rotated_overlays.append((rotate_affine(overlay_img, rot=canonical_r), params))
        overlay_images = rotated_overlays

    out_svgs = []
    for i, mode in enumerate(order):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        plot_params['title'] = label if i == 0 else None
        display = plot_anat(anat_nii, **plot_params)

        if ribbon:
            kwargs = {'levels': [0.5], 'linewidths': 0.5}
            display.add_contours(white, colors='b', **kwargs)
            display.add_contours(pial, colors='r', **kwargs)
        elif contour is not None:
            display.add_contours(contour, colors='r', levels=[0.5], linewidths=0.5)

        for overlay_img, params in overlay_images:
            display.add_overlay(overlay_img, **params)

        svg = extract_svg(display, compress=compress)
        display.close()
        svg = svg.replace('figure_1', f'{div_id}-{mode}-{uuid4()}', 1)
        out_svgs.append(svgt.fromstring(svg))

    return out_svgs


def _blend_with_white(rgba, blend=0.5):
    """Return a muted variant of the RGBA color by mixing with white."""
    rgb = np.array(rgba[:3])
    muted_rgb = (1 - blend) * rgb + blend
    return (*muted_rgb.tolist(), rgba[3])


def get_world_pedir(orientation: str, pe_dir: str) -> str:
    """Return the world phase-encoding direction."""

    orientation = orientation.upper()
    axis = pe_dir[0].lower()
    idx = {'i': 0, 'j': 1, 'k': 2}[axis]
    letter = orientation[idx]

    if pe_dir.endswith('-'):
        start = letter
        end = _OPPOSITE[letter]
    else:
        start = _OPPOSITE[letter]
        end = letter

    return f'{_ORI_TO_NAME[start]}-{_ORI_TO_NAME[end]}'


SUBJECT_TEMPLATE = """\
\t<ul class="elem-desc">
\t\t<li>Subject ID: {subject_id}</li>
\t\t<li>Structural images: {n_t1s:d} T1-weighted {t2w}</li>
\t\t<li>Functional series: {n_pet:d}</li>
{tasks}
\t\t<li>Standard output spaces: {std_spaces}</li>
\t\t<li>Non-standard output spaces: {nstd_spaces}</li>
\t\t<li>FreeSurfer reconstruction: {freesurfer_status}</li>
\t</ul>
"""

FUNCTIONAL_TEMPLATE = """\
\t\t<details open>
\t\t<summary>Summary</summary>
\t\t<ul class="elem-desc">
\t\t\t<li>Original orientation: {ornt}</li>
\t\t\t<li>Registration: {registration}</li>
\t\t\t<li>Anatomical reference: {anat_reference}</li>
\t\t\t<li>Reference image: {reference}</li>
\t\t\t<li>Time zero: {time_zero}</li>
\t\t\t<li>Radiotracer: {radiotracer}</li>
\t\t\t<li>Injected dose: {dose} {dose_units}</li>
\t\t\t<li>Scan duration: {duration} minutes</li>
\t\t\t<li>Number of frames: {n_frames}</li>
\t\t\t<li>Frame start times (seconds): {frame_start_times}</li>
\t\t\t<li>Frame durations (seconds): {frame_durations}</li>
\t\t</ul>
\t\t</details>
"""

ABOUT_TEMPLATE = """\t<ul>
\t\t<li>PETPrep version: {version}</li>
\t\t<li>PETPrep command: <code>{command}</code></li>
\t\t<li>Date preprocessed: {date}</li>
\t</ul>
</div>
"""


class SummaryOutputSpec(TraitedSpec):
    out_report = File(exists=True, desc='HTML segment containing summary')


class SummaryInterface(SimpleInterface):
    output_spec = SummaryOutputSpec

    def _run_interface(self, runtime):
        segment = self._generate_segment()
        fname = os.path.join(runtime.cwd, 'report.html')
        with open(fname, 'w') as fobj:
            fobj.write(segment)
        self._results['out_report'] = fname
        return runtime

    def _generate_segment(self):
        raise NotImplementedError


class SubjectSummaryInputSpec(BaseInterfaceInputSpec):
    t1w = InputMultiObject(File(exists=True), desc='T1w structural images')
    t2w = InputMultiObject(File(exists=True), desc='T2w structural images')
    subjects_dir = Directory(desc='FreeSurfer subjects directory')
    subject_id = Str(desc='Subject ID')
    pet = InputMultiObject(
        traits.Either(File(exists=True), traits.List(File(exists=True))),
        desc='PET functional series',
    )
    std_spaces = traits.List(Str, desc='list of standard spaces')
    nstd_spaces = traits.List(Str, desc='list of non-standard spaces')


class SubjectSummaryOutputSpec(SummaryOutputSpec):
    # This exists to ensure that the summary is run prior to the first ReconAll
    # call, allowing a determination whether there is a pre-existing directory
    subject_id = Str(desc='FreeSurfer subject ID')


class SubjectSummary(SummaryInterface):
    input_spec = SubjectSummaryInputSpec
    output_spec = SubjectSummaryOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.subject_id):
            self._results['subject_id'] = self.inputs.subject_id
        return super()._run_interface(runtime)

    def _generate_segment(self):
        BIDS_NAME = re.compile(
            r'^(.*\/)?'
            '(?P<subject_id>sub-[a-zA-Z0-9]+)'
            '(_(?P<session_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?'
            '(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?'
            '(_(?P<run_id>run-[a-zA-Z0-9]+))?'
        )

        if not isdefined(self.inputs.subjects_dir):
            freesurfer_status = 'Not run'
        else:
            recon = ReconAll(
                subjects_dir=self.inputs.subjects_dir,
                subject_id='sub-' + self.inputs.subject_id,
                T1_files=self.inputs.t1w,
                flags='-noskullstrip',
            )
            if recon.cmdline.startswith('echo'):
                freesurfer_status = 'Pre-existing directory'
            else:
                freesurfer_status = 'Run by PETPrep'

        t2w_seg = ''
        if self.inputs.t2w:
            t2w_seg = f'(+ {len(self.inputs.t2w):d} T2-weighted)'

        # Add list of tasks with number of runs
        pet_series = self.inputs.pet or []

        counts = Counter(
            (BIDS_NAME.search(series).groupdict().get('task_id') or 'task-<none>')[5:]
            for series in pet_series
        )

        tasks = ''
        if counts:
            header = '\t\t<ul class="elem-desc">'
            footer = '\t\t</ul>'
            lines = [
                f'\t\t\t<li>Task: {task_id} ({n_runs:d} run{"" if n_runs == 1 else "s"})</li>'
                for task_id, n_runs in sorted(counts.items())
            ]
            tasks = '\n'.join([header] + lines + [footer])

        return SUBJECT_TEMPLATE.format(
            subject_id=self.inputs.subject_id,
            n_t1s=len(self.inputs.t1w),
            t2w=t2w_seg,
            n_pet=len(pet_series),
            tasks=tasks,
            std_spaces=', '.join(self.inputs.std_spaces),
            nstd_spaces=', '.join(self.inputs.nstd_spaces),
            freesurfer_status=freesurfer_status,
        )


class FunctionalSummaryInputSpec(TraitedSpec):
    registration = traits.Enum(
        'mri_coreg',
        'mri_robust_register',
        'ants_registration',
        'auto_select',
        'Precomputed',
        mandatory=True,
        desc='PET/anatomical registration method',
    )
    registration_winner = traits.Enum(
        None,
        'ants',
        'freesurfer',
        allow_none=True,
        desc='Winner selected during automatic PET-to-T1w registration',
    )
    registration_dof = traits.Enum(
        6, 9, 12, desc='Registration degrees of freedom', mandatory=True
    )
    orientation = traits.Str(mandatory=True, desc='Orientation of the voxel axes')
    metadata = traits.Dict(desc='PET metadata dictionary')
    anatref_strategy = traits.Enum(
        't1w', 'nu', 'auto', desc='Anatomical reference used for registration'
    )
    requested_anatref = traits.Enum(
        None, 't1w', 'nu', 'auto', allow_none=True, desc='Requested anatomical reference'
    )
    volume_ratio = traits.Either(
        None,
        traits.Float(),
        usedefault=True,
        desc='PET-to-T1w mask volume ratio used for anatref auto-selection',
    )
    petref_strategy = traits.Enum(
        'template',
        'twa',
        'sum',
        'first5min',
        'auto',
        mandatory=True,
        desc='PET reference generation strategy',
    )
    requested_petref_strategy = traits.Enum(
        'template',
        'twa',
        'sum',
        'first5min',
        'auto',
        desc='User-requested PET reference strategy',
    )
    hmc_disabled = traits.Bool(False, desc='Head motion correction disabled')


class FunctionalSummary(SummaryInterface):
    input_spec = FunctionalSummaryInputSpec

    def _generate_segment(self):
        dof = self.inputs.registration_dof
        # TODO: Add a note about registration_init below?
        if self.inputs.registration == 'Precomputed':
            reg = 'Precomputed affine transformation'
        elif self.inputs.registration == 'mri_coreg':
            reg = f'FreeSurfer <code>mri_coreg</code> - {dof} dof'
        elif self.inputs.registration == 'ants_registration':
            reg = f'ANTs <code>ants_registration</code> ({dof} DoF)'
        elif self.inputs.registration == 'mri_robust_register':
            reg = 'FreeSurfer <code>mri_robust_register</code> (NMI cost)'
        elif self.inputs.registration == 'auto_select':
            winner = self.inputs.registration_winner
            if winner == 'ants':
                winner_desc = 'ANTs'
            elif winner == 'freesurfer':
                winner_desc = 'FreeSurfer'
            else:
                winner_desc = 'not recorded'
            reg = f'Automatic selection between FreeSurfer and ANTs (best score: {winner_desc})'
        else:
            reg = f'Unknown registration method: {self.inputs.registration}'

        anat_map = {
            't1w': 'Preprocessed T1w image',
            'nu': 'FreeSurfer bias-corrected volume (nu.mgz)',
            'auto': 'Automatically selected anatomical reference',
        }
        anat_reference = anat_map.get(self.inputs.anatref_strategy, 'Unknown')
        requested_anat = getattr(self.inputs, 'requested_anatref', None)
        volume_ratio = getattr(self.inputs, 'volume_ratio', None)
        if requested_anat == 'auto' and volume_ratio is not None:
            anat_reference += f' (PET/T1w mask volume ratio: {volume_ratio:.2f})'
        if requested_anat and requested_anat != self.inputs.anatref_strategy:
            anat_reference += f" (requested '{requested_anat}')"

        reference_map = {
            'template': 'Motion correction template',
            'twa': 'Time-weighted average of motion-corrected series',
            'sum': 'Summed motion-corrected series',
            'first5min': 'Early (0-5 minute) average of motion-corrected series',
            'auto': 'Automatically selected reference',
        }
        petref_strategy = reference_map.get(self.inputs.petref_strategy, 'Unknown')
        requested = getattr(self.inputs, 'requested_petref_strategy', None)
        if requested and requested != self.inputs.petref_strategy:
            petref_strategy += f" (requested '{requested}')"
        if self.inputs.hmc_disabled:
            petref_strategy += ' (head motion correction disabled)'

        meta = self.inputs.metadata or {}
        time_zero = meta.get('TimeZero', None)
        radiotracer = meta.get('TracerName')
        tracer_radionuclide = meta.get('TracerRadionuclide')
        if radiotracer and tracer_radionuclide:
            tracer_desc = f'{tracer_radionuclide}{radiotracer}'
        else:
            tracer_desc = 'n/a'
        dose = meta.get('InjectedRadioactivity')
        dose_units = meta.get('InjectedRadioactivityUnits', '')
        frame_times = meta.get('FrameTimesStart')
        frame_durations = meta.get('FrameDuration')
        n_frames = None
        duration = None
        if isinstance(frame_times, list):
            n_frames = len(frame_times)
            if isinstance(frame_durations, list):
                duration = frame_times[-1] + frame_durations[-1]
            elif frame_durations is not None:
                duration = frame_times[-1] + frame_durations
        elif isinstance(frame_durations, list):
            n_frames = len(frame_durations)
            duration = sum(frame_durations)
        elif frame_durations is not None:
            duration = frame_durations

        if duration is not None:
            duration = duration / 60.0  # Convert to minutes
            duration = f'{duration:.1f}'
        else:
            duration = 'n/a'

        dose_str = f'{dose}' if dose is not None else 'n/a'

        return FUNCTIONAL_TEMPLATE.format(
            registration=reg,
            anat_reference=anat_reference,
            reference=petref_strategy,
            ornt=self.inputs.orientation,
            # Use the metadata dictionary to fill in the details
            time_zero=time_zero,
            radiotracer=tracer_desc,
            dose=dose_str,
            dose_units=dose_units,
            duration=duration,
            n_frames=n_frames,
            frame_start_times=frame_times,
            frame_durations=frame_durations,
        )


class AboutSummaryInputSpec(BaseInterfaceInputSpec):
    version = Str(desc='PETPREP version')
    command = Str(desc='PETPREP command')
    # Date not included - update timestamp only if version or command changes


class AboutSummary(SummaryInterface):
    input_spec = AboutSummaryInputSpec

    def _generate_segment(self):
        return ABOUT_TEMPLATE.format(
            version=self.inputs.version,
            command=self.inputs.command,
            date=time.strftime('%Y-%m-%d %H:%M:%S %z'),
        )


class _AtlasROIsReportInputSpec(BaseInterfaceInputSpec):
    t1w_image = File(exists=True, mandatory=True, desc='Anatomical image resampled to PET space')
    petref_image = File(exists=True, mandatory=True, desc='PET reference image')
    segmentation = File(exists=True, mandatory=True, desc='Atlas segmentation in PET space')
    dseg_tsv = File(exists=True, mandatory=True, desc='Atlas label lookup table')
    atlas_name = Str(desc='Atlas name used for labeling')


class _AtlasROIsReportOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='SVG plot showing atlas regions')


class AtlasROIsReport(SimpleInterface):
    """Generate a figure showing atlas regions overlaid on T1w and PET images."""

    input_spec = _AtlasROIsReportInputSpec
    output_spec = _AtlasROIsReportOutputSpec

    def _run_interface(self, runtime):
        import matplotlib
        import pandas as pd

        matplotlib.use('Agg', force=True)
        from matplotlib import cm
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        atlas_labels = pd.read_csv(self.inputs.dseg_tsv, sep='\t')
        label_ids = [int(idx) for idx in atlas_labels.iloc[:, 0].tolist()]
        label_names = atlas_labels.iloc[:, 1].tolist() if len(atlas_labels.columns) > 1 else label_ids
        label_lookup = dict(zip(label_ids, map(str, label_names), strict=False))

        t1w_img = nb.load(self.inputs.t1w_image)
        pet_img = nb.load(self.inputs.petref_image)
        seg_img = nb.load(self.inputs.segmentation)

        seg_data = np.rint(seg_img.get_fdata()).astype(int)
        present_labels = [
            label for label in sorted(np.unique(seg_data)) if label in label_lookup and label != 0
        ]

        overlay_data = np.zeros(seg_data.shape, dtype='int32')
        for idx, label in enumerate(present_labels, start=1):
            overlay_data[seg_data == label] = idx
        overlay_img = nb.Nifti1Image(overlay_data, seg_img.affine, seg_img.header)

        muting_cycle = 2
        n_base_colors = max(1, int(np.ceil(len(present_labels) / muting_cycle)))
        color_map = cm.get_cmap('gist_rainbow', n_base_colors)
        rgba_colors = [(0, 0, 0, 0)]
        legend_handles = []
        for idx, label in enumerate(present_labels):
            base_idx = min(n_base_colors - 1, idx // muting_cycle)
            rgba = color_map(base_idx)
            if idx % muting_cycle:
                rgba = _blend_with_white(rgba)
            rgba_colors.append((*rgba[:3], 0.7))
            legend_handles.append(
                Patch(
                    facecolor=rgba[:3],
                    edgecolor='none',
                    label=f'{label} - {label_lookup[label]}',
                )
            )
        cmap = ListedColormap(rgba_colors)

        if overlay_data.any():
            mask_img = nlimage.new_img_like(seg_img, (overlay_data > 0).astype(np.uint8))
        else:
            mask_img = nlimage.threshold_img(t1w_img, 1e-3)
        cuts = cuts_from_bbox(mask_img, cuts=7)

        overlay_params = {
            'image': overlay_img,
            'params': {
                'cmap': cmap,
                'alpha': 1.0,
                'vmin': 0,
                'vmax': len(rgba_colors) - 1,
            },
        }

        t1_svgs = _plot_registration_with_overlays(
            t1w_img,
            'atlas-t1',
            cuts=cuts,
            estimate_brightness=True,
            label='T1w',
            dismiss_affine=True,
            overlays=[overlay_params],
        )

        pet_svgs = _plot_registration_with_overlays(
            pet_img,
            'atlas-pet',
            cuts=cuts,
            estimate_brightness=True,
            label='PET',
            dismiss_affine=True,
            overlays=[overlay_params],
        )

        legend_svg = None
        if legend_handles:
            legend_cols = min(5, len(legend_handles))
            rows = int(np.ceil(len(legend_handles) / legend_cols))
            fig, ax = plt.subplots(figsize=(12, max(1.0, rows * 0.4)))
            ax.axis('off')
            ax.legend(
                handles=legend_handles,
                ncol=legend_cols,
                loc='center',
                frameon=False,
                fontsize=10,
            )
            buf = io.StringIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            legend_svg = svgt.fromstring(buf.getvalue())

        overlay_file = os.path.join(runtime.cwd, 'atlas_rois_overlay.svg')
        compose_view(pet_svgs, t1_svgs, out_file=overlay_file)

        with open(overlay_file) as fobj:
            overlay_text = fobj.read()

        def _extract_dims_from_text(svg_text):
            viewbox_match = re.search(r'viewBox="([^"]+)"', svg_text)
            if viewbox_match:
                _, _, width_box, height_box = viewbox_match.group(1).split()
                return float(width_box), float(height_box)
            height_match = re.search(r'height="([^"]+)"', svg_text)
            width_match = re.search(r'width="([^"]+)"', svg_text)
            width_val = float(width_match.group(1).replace('px', '')) if width_match else 0.0
            height_val = float(height_match.group(1).replace('px', '')) if height_match else 0.0
            return width_val, height_val

        width, height = _extract_dims_from_text(overlay_text)
        total_height = height

        if legend_svg:
            legend_text = legend_svg.to_str()
            if isinstance(legend_text, (bytes, bytearray)):
                legend_text = legend_text.decode('utf-8')
            legend_text = legend_text.split('\n', 1)[-1]
            inner_match = re.search(r'<svg[^>]*>(.*)</svg>', legend_text, re.S)
            legend_body = inner_match.group(1) if inner_match else legend_text
            legend_width, legend_height = _extract_dims_from_text(legend_text)
            scale = width / legend_width if legend_width else 1.0
            legend_group = (
                f'<g transform="translate(0,{height}) scale({scale})">{legend_body}</g>'
            )
            total_height += legend_height * scale
            overlay_text = overlay_text.replace('</svg>', f'{legend_group}</svg>')

        overlay_text = re.sub(
            r'(viewBox="0 0 [^"]+ )([0-9.]+)(")',
            lambda m: f'{m.group(1)}{total_height}{m.group(3)}',
            overlay_text,
            count=1,
        )
        overlay_text = re.sub(
            r'(height=")([^"]+)(")',
            lambda m: f'{m.group(1)}{total_height}{m.group(3)}',
            overlay_text,
            count=1,
        )

        out_file = os.path.join(runtime.cwd, 'atlas_rois.svg')
        with open(out_file, 'w') as fobj:
            fobj.write(overlay_text)
        os.unlink(overlay_file)

        self._results['out_file'] = out_file
        return runtime
