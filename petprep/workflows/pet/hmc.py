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

from nipype.interfaces import fsl, freesurfer as fs
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
import numpy as np
import nibabel as nb
import nitransforms as nt


def get_min_frame(in_files: list[str]) -> int:
    """Return index of the frame with minimum mean intensity."""
    means = []
    for f in in_files:
        data = nb.load(f).get_fdata(dtype='float32')
        means.append(np.mean(data))
    return int(np.argmin(means))


def update_list_frames(frames: list[str], idx: int) -> list[str]:
    """Move selected frame to the first position."""
    frames = list(frames)
    if 0 <= idx < len(frames):
        frame = frames.pop(idx)
        frames.insert(0, frame)
    return frames


def update_list_transforms(xforms: list[str], idx: int) -> list[str]:
    """Move selected transform to the first position."""
    xforms = list(xforms)
    if 0 <= idx < len(xforms):
        xfm = xforms.pop(idx)
        xforms.insert(0, xfm)
    return xforms


class _LTAList2ITKInputSpec(niu.BaseInterfaceInputSpec):
    in_xforms = niu.InputMultiObject(niu.File(exists=True), mandatory=True)
    in_reference = niu.File(exists=True, mandatory=True)
    in_source = niu.InputMultiObject(niu.File(exists=True), mandatory=True)


class _LTAList2ITKOutputSpec(niu.TraitedSpec):
    out_file = niu.File(desc='output ITK transform list')


class LTAList2ITK(niu.SimpleInterface):
    input_spec = _LTAList2ITKInputSpec
    output_spec = _LTAList2ITKOutputSpec

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
        out_file = runtime.cwd / 'lta2itk.txt'
        affarray.to_filename(str(out_file))
        self._results['out_file'] = str(out_file)
        return runtime


def init_pet_hmc_wf(
    mem_gb: float,
    omp_nthreads: int,
    *,
    fwhm: float = 10.0,
    start_time: float = 120.0,
    name: str = 'pet_hmc_wf',
):
    """
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
        Time point (in seconds) defining the reference frame for motion estimation
    name : :obj:`str`
        Name of workflow (default: ``pet_hmc_wf``)

    Inputs
    ------
    pet_file
        PET series NIfTI file
    raw_ref_image
        Reference image to which PET series is motion corrected

    Outputs
    -------
    xforms
        ITKTransform file aligning each volume to ``ref_image``

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
        niu.IdentityInterface(fields=['pet_file', 'raw_ref_image']), name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['xforms']), name='outputnode')

    # Split frames
    split = pe.Node(fs.MRIConvert(out_type='niigz', split=True), name='split_frames')

    # Smooth and threshold frames
    smooth = pe.MapNode(
        fsl.Smooth(fwhm=fwhm),
        name='smooth',
        iterfield=['in_file'],
    )
    thresh = pe.MapNode(fsl.maths.Threshold(thresh=0.0), name='thresh', iterfield=['in_file'])

    # Select reference frame
    get_ref = pe.Node(niu.Function(function=get_min_frame), name='get_min_frame')
    upd_frames = pe.Node(niu.Function(function=update_list_frames), name='update_list_frames')

    # Motion estimation
    robtemp = pe.Node(
        fs.RobustTemplate(auto_detect_sensitivity=True,
                          intensity_scaling=True,
                          average_metric="mean",
                          args=f"--cras",
                          num_threads=omp_nthreads),
        name='robust_template',
    )
    upd_xfm = pe.Node(niu.Function(function=update_list_transforms), name='update_list_transforms')

    # Convert to ITK
    lta2itk = pe.Node(LTAList2ITK(), name='lta2itk', mem_gb=0.05, n_procs=omp_nthreads)

    workflow.connect([
        (inputnode, split, [('pet_file', 'in_file')]),
        (split, smooth, [('out_file', 'in_file')]),
        (smooth, thresh, [('out_file', 'in_file')]),
        (thresh, get_ref, [('out_file', 'in_files')]),
        (split, upd_frames, [('out_file', 'frames')]),
        (get_ref, upd_frames, [('out', 'idx')]),
        (upd_frames, robtemp, [('out', 'in_files')]),
        (robtemp, upd_xfm, [('transform_outputs', 'xforms')]),
        (get_ref, upd_xfm, [('out', 'idx')]),
        (upd_xfm, lta2itk, [('out', 'in_xforms')]),
        (upd_frames, lta2itk, [('out', 'in_source')]),
        (robtemp, lta2itk, [('out_file', 'in_reference')]),
        (lta2itk, outputnode, [('out_file', 'xforms')]),
    ])  # fmt:skip

    return workflow
