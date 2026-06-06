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
"""
Head-Motion Estimation and Correction (HMC) of PET images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_pet_hmc_wf

"""

from collections.abc import Sequence
from pathlib import Path

import nibabel as nb
import nitransforms as nt
import numpy as np
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
)
from nipype.interfaces.utility import Select
from nipype.pipeline import engine as pe


def get_start_frame(
    durations,
    start_time,
    frame_starts=None,
) -> int:
    """Return the index of the first frame whose midpoint exceeds ``start_time``.

    Parameters
    ----------
    durations
        Sequence of frame durations in seconds.
    start_time
        Time in seconds defining the onset of motion estimation.
    frame_starts
        Optional sequence specifying the start time of each frame.
        If omitted, cumulative ``durations`` will be used.
    """

    import numpy as np

    if durations is None:
        return 0

    durations = np.asarray(durations, dtype=float)
    if durations.size == 0:
        return 0

    if frame_starts is None:
        midpoints = np.cumsum(durations) - durations / 2.0
    else:
        midpoints = np.asarray(frame_starts, dtype=float) + durations / 2.0

    idxs = np.where(midpoints > start_time)[0]
    return int(idxs[0]) if idxs.size > 0 else int(len(midpoints) - 1)


def update_list_transforms(xforms: list[str], idx: int) -> list[str]:
    """
    Left-pad `xforms` by repeating the first transform `idx` times at the beginning.
    """
    if not xforms:
        raise ValueError('The input xforms list cannot be empty.')

    padded_xforms = [xforms[0]] * idx + xforms
    return padded_xforms


def lta_list(in_file):
    lta_list = [ext.replace('.nii.gz', '.lta') for ext in in_file]
    return lta_list


def _find_highest_uptake_frame(in_files: list[str]) -> int:
    import nibabel as nb
    import numpy as np

    uptake = [np.sum(nb.load(f).get_fdata(dtype=np.float32)) for f in in_files]
    return int(np.argmax(uptake)) + 1


class _LTAList2ITKInputSpec(BaseInterfaceInputSpec):
    in_xforms = InputMultiObject(File(exists=True), mandatory=True)
    in_reference = File(exists=True, mandatory=True)
    in_source = InputMultiObject(File(exists=True), mandatory=True)


class _LTAList2ITKOutputSpec(TraitedSpec):
    out_file = File(desc='output ITK transform list')


class LTAList2ITK(SimpleInterface):
    input_spec = _LTAList2ITKInputSpec
    output_spec = _LTAList2ITKOutputSpec
    """Convert FreeSurfer LTA transforms into an ITK transform list."""

    def _run_interface(self, runtime):
        reference = nb.load(self.inputs.in_reference)
        sources = [nb.load(f) for f in self.inputs.in_source]
        affines = [
            nt.linear.load(xf, fmt='fs', reference=reference, moving=src)
            for xf, src in zip(self.inputs.in_xforms, sources, strict=False)
        ]
        affarray = nt.io.itk.ITKLinearTransformArray.from_ras(
            np.stack([a.matrix for a in affines], axis=0)
        )
        out_file = Path(runtime.cwd) / 'lta2itk.txt'
        affarray.to_filename(str(out_file))
        self._results['out_file'] = str(out_file)
        return runtime


def init_pet_hmc_wf(
    mem_gb: float,
    omp_nthreads: int,
    *,
    fwhm: float = 10.0,
    start_time: float = 120.0,
    frame_durations: Sequence[float] | None = None,
    frame_start_times: Sequence[float] | None = None,
    initial_frame: int | str | None = 'auto',
    fixed_frame: bool = False,
    name: str = 'pet_hmc_wf',
):
    r"""
    Build a workflow to estimate head-motion parameters.

    This workflow estimates the motion parameters to perform
    :abbr:`HMC (head motion correction)` over the input
    :abbr:`PET (positron emission tomography)` image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.pet import init_pet_hmc_wf
            wf = init_pet_hmc_wf(
                mem_gb=3,
                omp_nthreads=1,
                fwhm=10.0,
                start_time=120.0,
            )

    Parameters
    ----------
    mem_gb : :obj:`float`
        Size of PET file in GB
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    fwhm : :obj:`float`
        FWHM in millimeters for Gaussian smoothing prior to motion estimation
    start_time : :obj:`float`
        Earliest time point (in seconds) used for motion estimation.
    frame_durations : :class:`~typing.Sequence`\[:obj:`float`] or ``None``
        Duration of each frame in seconds. If not provided, start-time clamping
        will be skipped.
    frame_start_times : :class:`~typing.Sequence`\[:obj:`float`] or ``None``
        Optional list of frame onset times used together with
        ``frame_durations`` to locate the start frame.
    initial_frame : :obj:`int`, ``'auto'`` or ``None``
        0-based index of the frame used to initialize motion correction. If ``'auto'`` or
        ``None`` (default), the frame with the highest uptake is selected automatically.
        FreeSurfer's ``initial_timepoint`` is 1-based; this workflow applies the
        required offset internally.
    fixed_frame : :obj:`bool`
        Whether to keep the initial time point fixed during robust template
        estimation (``fs.RobustTemplate``'s ``fixtp`` parameter). If ``True``,
        iterations are skipped to reduce runtime.
    name : :obj:`str`
        Name of workflow (default: ``pet_hmc_wf``)

    Inputs
    ------
    pet_file
        PET series NIfTI file
    frame_durations
        Duration of each PET frame, in seconds.
    frame_start_times
        Optional onset time of each PET frame.

    Outputs
    -------
    xforms
        ITKTransform file aligning each volume to ``ref_image``
    petref
        PET reference image generated during motion estimation

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Head-motion parameters with respect to the PET reference
(transformation matrices, and six corresponding rotation and translation
parameters) are estimated before any spatiotemporal filtering using
FreeSurfer's ``mri_robust_template``.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['pet_file', 'start_time', 'frame_durations', 'frame_start_times']
        ),
        name='inputnode',
    )
    inputnode.inputs.start_time = start_time
    inputnode.inputs.frame_durations = frame_durations
    inputnode.inputs.frame_start_times = frame_start_times
    outputnode = pe.Node(niu.IdentityInterface(fields=['xforms', 'petref']), name='outputnode')

    # Split frames
    split = pe.Node(fs.MRIConvert(out_type='niigz', split=True), name='split_frames')

    # After splitting, explicitly select frames
    select_frames = pe.Node(Select(), name='select_frames')

    # Define a function to create the correct indices
    def get_frame_indices(total_frames, start_idx):
        return list(range(start_idx, total_frames))

    # Explicit function for Nipype connection
    def num_files(filelist):
        return len(filelist)

    num_files_node = pe.Node(
        niu.Function(input_names=['filelist'], output_names=['length'], function=num_files),
        name='num_files_node',
    )

    select_idx = pe.Node(
        niu.Function(
            input_names=['total_frames', 'start_idx'],
            output_names=['indices'],
            function=get_frame_indices,
        ),
        name='select_indices',
    )

    # Smooth and threshold frames
    smooth = pe.MapNode(
        fsl.Smooth(fwhm=fwhm),
        name='smooth',
        iterfield=['in_file'],
    )
    thresh = pe.MapNode(
        fsl.maths.Threshold(thresh=20, use_robust_range=True), name='thresh', iterfield=['in_file']
    )

    # Select reference frame
    start_frame = pe.Node(
        niu.Function(
            input_names=['durations', 'start_time', 'frame_starts'],
            output_names=['start_frame_idx'],
            function=get_start_frame,
        ),
        name='get_start_frame',
    )
    start_frame.inputs.start_time = start_time

    make_lta_list = pe.Node(
        niu.Function(
            input_names=['in_file'],
            output_names=['lta_list'],
            function=lta_list,
        ),
        name='create_lta_list',
    )

    auto_init_frame = initial_frame in (None, 'auto')
    if auto_init_frame:
        find_highest_uptake_frame = pe.Node(
            niu.Function(
                input_names=['in_files'],
                output_names=['index'],
                function=_find_highest_uptake_frame,
            ),
            name='find_highest_uptake_frame',
        )

    # Motion estimation
    robust_template = pe.Node(
        fs.RobustTemplate(
            auto_detect_sensitivity=True,
            intensity_scaling=True,
            average_metric='mean',
            args='--cras',
            num_threads=omp_nthreads,
            fixed_timepoint=fixed_frame,
            no_iteration=fixed_frame,
        ),
        name='est_robust_hmc',
    )
    if not auto_init_frame:
        robust_template.inputs.initial_timepoint = int(initial_frame) + 1
    upd_xfm = pe.Node(
        niu.Function(
            input_names=['xforms', 'idx'],
            output_names=['updated_xforms'],
            function=update_list_transforms,
        ),
        name='update_list_transforms',
    )

    # Convert to ITK
    lta2itk = pe.Node(LTAList2ITK(), name='convert_lta2itk', mem_gb=0.05, n_procs=omp_nthreads)

    ref_to_nii = pe.Node(fs.MRIConvert(out_type='niigz'), name='convert_ref')

    workflow.connect([
        (inputnode, split, [('pet_file', 'in_file')]),
        (inputnode, start_frame, [('frame_durations', 'durations'),
                                  ('frame_start_times', 'frame_starts'),
                                  ('start_time', 'start_time')]),
        (split, num_files_node, [('out_file', 'filelist')]),
        (num_files_node, select_idx, [('length', 'total_frames')]),
        (start_frame, select_idx, [('start_frame_idx', 'start_idx')]),
        (select_idx, select_frames, [('indices', 'index')]),
        (split, select_frames, [('out_file', 'inlist')]),
        (select_frames, smooth, [('out', 'in_file')]),
        (smooth, thresh, [('smoothed_file', 'in_file')]),
        (thresh, make_lta_list, [('out_file', 'in_file')]),
        (make_lta_list, robust_template, [('lta_list', 'transform_outputs')]),
        (thresh, robust_template, [('out_file', 'in_files')]),
        (robust_template, upd_xfm, [('transform_outputs', 'xforms')]),
        (start_frame, upd_xfm, [('start_frame_idx', 'idx')]),
        (upd_xfm, lta2itk, [('updated_xforms', 'in_xforms')]),
        (robust_template, lta2itk, [('out_file', 'in_reference')]),
        (split, lta2itk, [('out_file', 'in_source')]),
        (lta2itk, outputnode, [('out_file', 'xforms')]),
        (robust_template, ref_to_nii, [('out_file', 'in_file')]),
        (ref_to_nii, outputnode, [('out_file', 'petref')]),
    ])  # fmt:skip

    if auto_init_frame:
        workflow.connect(
            [
                (thresh, find_highest_uptake_frame, [('out_file', 'in_files')]),
                (find_highest_uptake_frame, robust_template, [('index', 'initial_timepoint')]),
            ]
        )

    return workflow
