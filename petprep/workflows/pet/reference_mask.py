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
"""Workflows for creation of reference masks."""

from __future__ import annotations

import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu

from petprep import config
from petprep.interfaces import DerivativesDataSink
from petprep.interfaces.reference_mask import ExtractRefRegion
from petprep.utils.reference_mask import mask_to_stats


def init_pet_refmask_wf(
    *,
    segmentation: str,
    ref_mask_name: str,
    ref_mask_index: list[int] | None = None,
    config_path: str,
    name: str = 'pet_refmask_wf',
) -> pe.Workflow:
    import nipype.pipeline.engine as pe
    from nipype.interfaces.utility import IdentityInterface

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        IdentityInterface(fields=['seg_file', 'gm_probseg']),
        name='inputnode',
    )
    outputnode = pe.Node(
        IdentityInterface(fields=['refmask_file', 'morph_tsv']), name='outputnode'
    )

    extract_mask = pe.Node(ExtractRefRegion(), name='extract_refregion')
    extract_mask.inputs.segmentation_type = segmentation
    extract_mask.inputs.region_name = ref_mask_name
    extract_mask.inputs.config_file = config_path

    make_morph = pe.Node(
        niu.Function(
            input_names=['mask_file', 'mask_name'],
            output_names=['out_file'],
            function=mask_to_stats,
        ),
        name='make_morphtsv',
    )
    make_morph.inputs.mask_name = ref_mask_name

    ds_morph_tsv = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            label=ref_mask_name,
            desc='ref',
            allowed_entities=('label',),
            suffix='morph',
            extension='.tsv',
            datatype='anat',
            check_hdr=False,
        ),
        name='ds_morphtsv',
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    if ref_mask_index is not None:
        # Override config-based lookup and force manual indices
        extract_mask.inputs.override_indices = ref_mask_index

    workflow.connect(
        [
            (
                inputnode,
                extract_mask,
                [
                    ('seg_file', 'seg_file'),
                    ('gm_probseg', 'gm_probseg'),
                ],
            ),
            (
                extract_mask,
                make_morph,
                [('refmask_file', 'mask_file')],
            ),
            (
                make_morph,
                ds_morph_tsv,
                [('out_file', 'in_file')],
            ),
            (
                inputnode,
                ds_morph_tsv,
                [('seg_file', 'source_file')],
            ),
            (make_morph, outputnode, [('out_file', 'morph_tsv')]),
            (extract_mask, outputnode, [('refmask_file', 'refmask_file')]),
        ]
    )

    return workflow
