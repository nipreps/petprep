from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer.petsurfer import GTMPVC
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nitransforms import ConcatenateXFMs

from ... import config
from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink


def init_gtmpvc_reg_wf(name: str = "gtmpvc_reg_wf") -> Workflow:
    """Create a registration transform for ``mri_gtmpvc``."""
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["motion_xfm", "petref2anat_xfm", "pet_ref", "t1w_preproc"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["reg_lta"]), name="outputnode")

    merge_xfms = pe.Node(niu.Merge(2), name="merge_xfms", run_without_submitting=True)
    concat_xfm = pe.Node(
        ConcatenateXFMs(out_fmt="fs", inverse=True),
        name="concat_xfm",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
        [
            (
                inputnode,
                merge_xfms,
                [("motion_xfm", "in1"), ("petref2anat_xfm", "in2")],
            ),
            (merge_xfms, concat_xfm, [("out", "in_xfms")]),
            (
                inputnode,
                concat_xfm,
                [("t1w_preproc", "reference"), ("pet_ref", "moving")],
            ),
            (concat_xfm, outputnode, [("out_inv", "reg_lta")]),
        ]
    )

    return workflow


def init_gtmpvc_wf(*, metadata: dict, name: str = "gtmpvc_wf") -> Workflow:
    """Run FreeSurfer ``mri_gtmpvc``."""
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["pet", "segmentation", "reg_lta"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["pvc_pet"]), name="outputnode")

    gtmpvc = pe.Node(GTMPVC(), name="gtmpvc")
    ds_pvc = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            desc="gtmpvc",
            suffix="pet",
            datatype="pet",
            compress=True,
            TaskName=metadata.get("TaskName"),
        ),
        name="ds_gtmpvc",
        run_without_submitting=True,
        mem_gb=DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect(
        [
            (
                inputnode,
                gtmpvc,
                [
                    ("pet", "in_file"),
                    ("segmentation", "segmentation"),
                    ("reg_lta", "reg_file"),
                ],
            ),
            (gtmpvc, ds_pvc, [("gtm_file", "in_file")]),
            (inputnode, ds_pvc, [("pet", "source_file")]),
            (ds_pvc, outputnode, [("out_file", "pvc_pet")]),
        ]
    )

    return workflow