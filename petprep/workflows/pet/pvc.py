from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.freesurfer.petsurfer import GTMPVC
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.nitransforms import ConcatenateXFMs

from ... import config
from ...config import DEFAULT_MEMORY_MIN_GB
from ...interfaces import DerivativesDataSink

class LoggingConcatenateXFMs(ConcatenateXFMs):
        def _run_interface(self, runtime):
            try:
                return super()._run_interface(runtime)
            except TypeError:
                config.loggers.workflow.error(
                    "Failed to concatenate transforms:\n%s",
                    "\n".join(self.inputs.in_xfms),
                )
                raise


def init_gtmpvc_reg_wf(name: str = "gtmpvc_reg_wf") -> Workflow:
    """Create a registration transform for ``mri_gtmpvc``.

    The motion-correction and registration transforms are concatenated
    with :class:`niworkflows.interfaces.nitransforms.ConcatenateXFMs`. If
    that step fails with a ``TypeError`` the paths to the transforms being
    merged are logged to help locate the issue.
    """
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
        LoggingConcatenateXFMs(out_fmt="fs", inverse=True),
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


def init_gtmpvc_wf(*, metadata: dict, method: str, name: str = "gtmpvc_wf") -> Workflow:
    """Run FreeSurfer ``mri_gtmpvc``.

    Parameters
    ----------
    metadata : dict
        BIDS metadata dictionary associated with the PET series.
    method : {"gtm", "mg", "rbv"}
        Partial volume correction method.
    name : str
        Workflow name.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["pet", "segmentation", "reg_lta"]),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["pvc_pet"]), name="outputnode")

    pvc_dir = f"pvc-{method}"
    gtmpvc_opts = {
        "default_seg_merge": True,
        "auto_mask": (1, 0.1),
        "pvc_dir": pvc_dir,
        "no_rescale": True,
        "no_reduce_fov": True,
        "no_pvc": False,
    }
    if method == 'rbv':
        gtmpvc_opts['rbv'] = True
    elif method == 'mg':
        gtmpvc_opts['mgx'] = 0.0

    gtmpvc = pe.Node(GTMPVC(**gtmpvc_opts), name='gtmpvc')

    if getattr(config.workflow, 'pvc_psf', None):
        psf = config.workflow.pvc_psf
        if isinstance(psf, (list | tuple)) and len(psf) == 4:
            (
                gtmpvc.inputs.psf_col,
                gtmpvc.inputs.psf_row,
                gtmpvc.inputs.psf_slice,
            ) = psf
        else:
            gtmpvc.inputs.psf = psf

    ds_pvc = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.petprep_dir,
            desc=f"pvc-{method}",
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
            (
                gtmpvc,
                ds_pvc,
                [
                    (
                        {"gtm": "gtm_file", "mg": "mgx_gm", "rbv": "rbv"}[method],
                        "in_file",
                    )
                ],
            ),
            (inputnode, ds_pvc, [("pet", "source_file")]),
            (ds_pvc, outputnode, [("out_file", "pvc_pet")]),
        ]
    )

    return workflow
