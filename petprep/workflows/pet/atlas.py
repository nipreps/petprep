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
"""Atlas workflows."""

from __future__ import annotations

import json

from nipype.interfaces import utility as niu
from nipype.interfaces.ants import Registration
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms

from ... import config
from ...interfaces import DerivativesDataSink
from ...interfaces.bids import BIDSURI
from templateflow.api import get as get_template

DEFAULT_MEMORY_MIN_GB = config.DEFAULT_MEMORY_MIN_GB


def _atlas_morph_tsv(segmentation: str, labels_tsv: str) -> str:
    """Generate a TSV table of region volumes from a segmentation."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    import pandas as pd

    seg_img = nb.load(segmentation)
    data = np.asanyarray(seg_img.dataobj)
    voxvol = np.prod(seg_img.header.get_zooms()[:3])
    labels = pd.read_csv(labels_tsv, sep="\t")
    volumes = [(data == int(idx)).sum() * voxvol for idx in labels["index"]]
    out = labels.copy()
    out["volume-mm3"] = volumes
    out_file = Path("morph.tsv").absolute()
    out.to_csv(out_file, sep="\t", index=False)
    return str(out_file)


def init_atlas_wf(atlas: str, config_file: str, name: str = "pet_atlas_wf") -> Workflow:
    """Map a template atlas into T1w space and compute regional volumes.

    Parameters
    ----------
    atlas : :class:`str`
        Name of atlas (used in outputs and TemplateFlow queries).
    config_file : :class:`str`
        JSON file with query parameters for TemplateFlow ``get`` calls.
    name : :class:`str`
        Workflow name (default: ``pet_atlas_wf``).

    Inputs
    ------
    t1w_preproc
        Preprocessed T1-weighted image.

    Outputs
    -------
    segmentation
        Atlas segmentation in T1w space.
    dseg_tsv
        Label table for the atlas.
    """
    with open(config_file) as f:
       data = json.load(f)

    if atlas not in data:
        raise ValueError(
            f"Atlas '{atlas}' not found in {config_file}. "
            f"Available atlases: {', '.join(sorted(data.keys()))}"
        )

    cfg = data[atlas]
    required = {"t1w", "atlas", "labels"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(
            f"Atlas '{atlas}' missing required keys: {', '.join(sorted(missing))}"
        )

    def _tf_kwargs(d: dict) -> dict:
        out = dict(d)
        if "tpl" in out:
            out["template"] = out.pop("tpl")
        if "res" in out:
            out["resolution"] = out.pop("res")
        return out

    template_t1w = str(get_template(**_tf_kwargs(cfg["t1w"])))
    atlas_img = str(get_template(**_tf_kwargs(cfg["atlas"])))
    labels_tsv = str(get_template(**_tf_kwargs(cfg["labels"])))

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=["t1w_preproc"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["segmentation", "dseg_tsv"]),
        name="outputnode",
    )

    label_source = pe.Node(niu.IdentityInterface(fields=["dseg_tsv"]), name="label_source")
    label_source.inputs.dseg_tsv = labels_tsv

    reg = pe.Node(
        Registration(
            transforms=['Rigid', 'Affine', 'SyN'],
            transform_parameters=[(0.1,), (0.1,), (0.1, 3, 0)],
            metric=['Mattes', 'Mattes', 'CC'],
            metric_weight=[1, 1, 1],
            radius_or_number_of_bins=[32, 32, 4],
            sampling_strategy=['Regular', 'Regular', None],
            sampling_percentage=[0.25, 0.25, None],
            sigma_units=['vox', 'vox', 'vox'],
            number_of_iterations=[
                [1000, 500, 250, 0],
                [1000, 500, 250, 0],
                [100, 70, 50, 10],
            ],
            shrink_factors=[
                [8, 4, 2, 1],
                [8, 4, 2, 1],
                [8, 4, 2, 1],
            ],
            smoothing_sigmas=[
                [3, 2, 1, 0],
                [3, 2, 1, 0],
                [3, 2, 1, 0],
            ],
            use_histogram_matching=True,
            write_composite_transform=True,
        ),
        name="t1_to_tpl",
    )
    reg.inputs.fixed_image = template_t1w

    apply_inv = pe.Node(ApplyTransforms(interpolation="NearestNeighbor"), name="apply_atlas")
    apply_inv.inputs.input_image = atlas_img

    gen_morph = pe.Node(
        niu.Function(
            input_names=["segmentation", "labels_tsv"],
            output_names=["out_file"],
            function=_atlas_morph_tsv,
        ),
        name="gen_morph",
    )
    gen_morph.inputs.labels_tsv = labels_tsv

    sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.petprep_dir),
        ),
        name="sources",
    )

    ds_seg = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            seg=atlas,
            allowed_entities=("seg",),
            suffix="dseg",
            extension=".nii.gz",
            compress=True,
        ),
        name="ds_seg",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    ds_dseg_tsv = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            seg=atlas,
            allowed_entities=("seg",),
            suffix="dseg",
            extension=".tsv",
            datatype="anat",
            check_hdr=False,
        ),
        name="ds_dseg_tsv",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    ds_morph_tsv = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            seg=atlas,
            allowed_entities=("seg",),
            suffix="morph",
            extension=".tsv",
            datatype="anat",
            check_hdr=False,
        ),
        name="ds_morph_tsv",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
        [
            (inputnode, reg, [("t1w_preproc", "moving_image")]),
            (reg, apply_inv, [("inverse_composite_transform", "transforms")]),
            (inputnode, apply_inv, [("t1w_preproc", "reference_image")]),
            (apply_inv, outputnode, [("output_image", "segmentation")]),
            (label_source, outputnode, [("dseg_tsv", "dseg_tsv")]),
            (inputnode, sources, [("t1w_preproc", "in1")]),
            (apply_inv, ds_seg, [("output_image", "in_file")]),
            (inputnode, ds_seg, [("t1w_preproc", "source_file")]),
            (sources, ds_seg, [("out", "Sources")]),
            (label_source, ds_dseg_tsv, [("dseg_tsv", "in_file")]),
            (inputnode, ds_dseg_tsv, [("t1w_preproc", "source_file")]),
            (sources, ds_dseg_tsv, [("out", "Sources")]),
            (apply_inv, gen_morph, [("output_image", "segmentation")]),
            (gen_morph, ds_morph_tsv, [("out_file", "in_file")]),
            (inputnode, ds_morph_tsv, [("t1w_preproc", "source_file")]),
            (sources, ds_morph_tsv, [("out", "Sources")]),
        ]
    )  # fmt:skip

    return workflow
