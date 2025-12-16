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
from smriprep.interfaces.freesurfer import ReconAll

try:  # NiReports >= 24.1 vendors svgutils
    import nireports._vendored.svgutils.transform as svgt
except ImportError:  # Fall back to system svgutils for older NiReports releases
    import svgutils.transform as svgt
from nireports.reportlets.utils import compose_view, cuts_from_bbox, extract_svg, robust_set_limits
from nireports.tools.ndimage import rotate_affine, rotation2canonical

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
        'Precomputed',
        mandatory=True,
        desc='PET/anatomical registration method',
    )
    registration_dof = traits.Enum(
        6, 9, 12, desc='Registration degrees of freedom', mandatory=True
    )
    orientation = traits.Str(mandatory=True, desc='Orientation of the voxel axes')
    metadata = traits.Dict(desc='PET metadata dictionary')


class FunctionalSummary(SummaryInterface):
    input_spec = FunctionalSummaryInputSpec

    def _generate_segment(self):
        dof = self.inputs.registration_dof
        # TODO: Add a note about registration_init below?
        if self.inputs.registration == 'Precomputed':
            reg = 'Precomputed affine transformation'
        elif self.inputs.registration == 'mri_coreg':
            reg = f'FreeSurfer <code>mri_coreg</code> - {dof} dof'
        else:
            reg = 'FreeSurfer <code>mri_robust_register</code> (ROBENT cost)'

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
        import numpy as np
        import pandas as pd
        import nibabel as nb
        from nilearn import image as nlimage
        from nilearn.plotting import plot_anat
        import matplotlib

        matplotlib.use('Agg', force=True)
        from matplotlib import cm, pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        atlas_labels = pd.read_csv(self.inputs.dseg_tsv, sep='\t')
        label_ids = [int(idx) for idx in atlas_labels.iloc[:, 0].tolist()]
        label_names = atlas_labels.iloc[:, 1].tolist() if len(atlas_labels.columns) > 1 else label_ids
        label_lookup = dict(zip(label_ids, map(str, label_names), strict=False))

        t1w_img = nb.load(self.inputs.t1w_image)
        pet_img = nb.load(self.inputs.petref_image)
        seg_img = nb.load(self.inputs.segmentation)

        rotation = rotation2canonical(t1w_img)
        if rotation is not None:
            t1w_img = rotate_affine(t1w_img, rot=rotation)
            pet_img = rotate_affine(pet_img, rot=rotation)
            seg_img = rotate_affine(seg_img, rot=rotation)

        seg_data = np.rint(seg_img.get_fdata()).astype(int)

        mask_data = seg_data > 0
        if mask_data.any():
            mask_img = nlimage.new_img_like(seg_img, mask_data.astype(np.uint8))
        else:
            mask_img = nlimage.threshold_img(t1w_img, 1e-3)

        cuts = cuts_from_bbox(mask_img, cuts=7)

        present_labels = [
            label for label in sorted(np.unique(seg_data)) if label in label_lookup and label != 0
        ]

        color_map = cm.get_cmap('tab20', max(len(present_labels), 1))
        overlay_specs = []
        for idx, label in enumerate(present_labels):
            mask = (seg_data == label).astype(np.uint8)
            if not mask.any():
                continue
            overlay_img = nb.Nifti1Image(mask, seg_img.affine)
            overlay_specs.append((overlay_img, color_map(idx), label))

        def _to_svg_element(svg_data):
            if isinstance(svg_data, bytes):
                text = svg_data.decode('utf-8')
            else:
                text = str(svg_data)
            return svgt.fromstring(text)

        def _plot_overlay(bg_img, title, plot_params):
            svgs = []
            for i, axis in enumerate(('z', 'x', 'y')):
                params = dict(plot_params)
                params['display_mode'] = axis
                params['cut_coords'] = cuts[axis]
                params['title'] = title if i == 0 else None
                display = plot_anat(bg_img, **params)
                for overlay_img, color, _ in overlay_specs:
                    cmap = ListedColormap([(0, 0, 0, 0), (*color[:3], 0.7)])
                    display.add_overlay(
                        overlay_img,
                        cmap=cmap,
                        alpha=1.0,
                        vmin=0,
                        vmax=1,
                    )
                svg = extract_svg(display, compress='auto')
                svgs.append(_to_svg_element(svg))
                display.close()
            return svgs

        t1_params = robust_set_limits(np.asanyarray(t1w_img.dataobj), {})
        pet_params = robust_set_limits(np.asanyarray(pet_img.dataobj), {})

        bg_svgs = []
        bg_svgs.extend(_plot_overlay(t1w_img, 'T1-weighted anatomical', t1_params))
        bg_svgs.extend(_plot_overlay(pet_img, 'PET reference', pet_params))

        legend_svg = None
        if overlay_specs:
            legend_cols = min(5, len(overlay_specs))
            rows = int(np.ceil(len(overlay_specs) / legend_cols))
            fig, ax = plt.subplots(figsize=(12, max(1.0, rows * 0.4)))
            ax.axis('off')
            handles = [
                Patch(
                    facecolor=color_map(idx)[:3],
                    edgecolor='none',
                    label=f"{label} - {label_lookup[label]}",
                )
                for idx, (_, _, label) in enumerate(overlay_specs)
            ]
            ax.legend(
                handles=handles,
                ncol=legend_cols,
                loc='center',
                frameon=False,
                fontsize=10,
            )
            buf = io.StringIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            legend_svg = _to_svg_element(buf.getvalue())

        out_file = os.path.join(runtime.cwd, 'atlas_rois.svg')
        compose_view(bg_svgs, [legend_svg] if legend_svg else [], out_file=out_file)

        self._results['out_file'] = out_file
        return runtime
