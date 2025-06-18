from __future__ import annotations

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe

from ...interfaces.pvc import Binarise4DSegmentation
from nipype.interfaces.petpvc import PETPVC
from nipype.interfaces.freesurfer.petsurfer import GTMPVC


def init_pet_pvc_wf(
    *,
    method: str | None = None,
    fwhm: tuple[float, float, float] | None = None,
    name: str = 'pet_pvc_wf',
) -> pe.Workflow:
    """Apply partial volume correction to a PET series."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=['pet_file', 'anat_seg']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['pet_file']), name='outputnode')

    pvc = pe.Node(PVC(method=method or 'none'), name='pvc')
    if fwhm is not None:
        pvc.inputs.fwhm = fwhm

    workflow.connect(
        [
            (inputnode, pvc, [('pet_file', 'in_file'), ('anat_seg', 'mask_file')]),
            (pvc, outputnode, [('out_file', 'pet_file')]),
        ]
    )

    return workflow
