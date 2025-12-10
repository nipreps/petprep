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
            reg = (
                'Automatic selection between FreeSurfer and ANTs '
                f'(best score: {winner_desc})'
            )
        else:
            reg = f'Unknown registration method: {self.inputs.registration}'

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
