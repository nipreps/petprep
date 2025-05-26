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
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.header import ValidateImage
from nipype.interfaces import fsl

DEFAULT_MEMORY_MIN_GB = 0.01


def init_raw_petref_wf(
    pet_file=None,
    *,
    reference_frame: int | str | None = None,
    name='raw_petref_wf',
):
    """
    Build a workflow that generates reference PET images for a series.

    The raw reference image is the target of :abbr:`HMC (head motion correction)`, and a
    contrast-enhanced reference is the subject of distortion correction, as well as
    boundary-based registration to T1w and template spaces.

    This workflow assumes only one PET file has been passed.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from petprep.workflows.pet.reference import init_raw_petref_wf
            wf = init_raw_petref_wf()

    Parameters
    ----------
    pet_file : :obj:`str`
        PET series NIfTI file
    reference_frame : :obj:`int` or ``"average"`` or ``None``
        Select a specific volume to use as reference. ``None`` or ``"average"``
        computes a robust average across frames.
    name : :obj:`str`
        Name of workflow (default: ``pet_reference_wf``)

    Inputs
    ------
    pet_file : str
        PET series NIfTI file

    Outputs
    -------
    pet_file : str
        Validated PET series NIfTI file
    petref : str
        Reference image to which PET series is motion corrected

    """
    from niworkflows.interfaces.images import RobustAverage

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
First, a reference volume was generated,
using a custom methodology of *fMRIPrep*, for use in head motion correction.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file']),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'pet_file',
                'petref',
                'validation_report',
            ]
        ),
        name='outputnode',
    )

    # Simplify manually setting input image
    if pet_file is not None:
        inputnode.inputs.pet_file = pet_file

    val_pet = pe.Node(ValidateImage(), name='val_pet', mem_gb=DEFAULT_MEMORY_MIN_GB)

    gen_avg = pe.Node(RobustAverage(), name='gen_avg', mem_gb=1)
    extract_roi = pe.Node(
        fsl.ExtractROI(t_size=1),
        name='extract_frame',
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
        [
            (inputnode, val_pet, [('pet_file', 'in_file')]),
            (val_pet, outputnode, [
                ('out_file', 'pet_file'),
                ('out_report', 'validation_report'),
            ]),
        ]
    )  # fmt:skip

    if reference_frame in (None, 'average'):
        workflow.connect(
            [
                (val_pet, gen_avg, [('out_file', 'in_file')]),
                (gen_avg, outputnode, [('out_file', 'petref')]),
            ]
        )  # fmt:skip
    else:
        extract_roi.inputs.t_min = int(reference_frame)
        workflow.connect(
            [
                (val_pet, extract_roi, [('out_file', 'in_file')]),
                (extract_roi, outputnode, [('roi_file', 'petref')]),
            ]
        )  # fmt:skip

    return workflow


def init_validation_and_dummies_wf(
    pet_file=None,
    name='validation_and_dummies_wf',
):
    """Build a workflow that validates a PET image."""

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['pet_file']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'validation_report']),
        name='outputnode',
    )

    if pet_file is not None:
        inputnode.inputs.pet_file = pet_file

    val_pet = pe.Node(ValidateImage(), name='val_pet', mem_gb=DEFAULT_MEMORY_MIN_GB)

    workflow.connect([
        (inputnode, val_pet, [('pet_file', 'in_file')]),
        (val_pet, outputnode, [
            ('out_file', 'pet_file'),
            ('out_report', 'validation_report'),
        ]),
    ])

    return workflow
