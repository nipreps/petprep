from __future__ import annotations

import json
from pathlib import Path
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.petpvc import PETPVC
from nipype.interfaces.freesurfer.petsurfer import GTMPVC


def load_pvc_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def init_pet_pvc_wf(
    *,
    tool: str = 'PETPVC',
    method: str = 'GTM',
    pvc_params: dict | None = None,
    config_path: Path,
    name: str = 'pet_pvc_wf',
) -> pe.Workflow:
    """Apply partial volume correction to a PET series using PETPVC or PETSurfer."""

    config = load_pvc_config(config_path)

    tool_lower = tool.lower()
    method_key = method.upper() if tool_lower == 'petpvc' else method.lower()

    if method_key not in config.get(tool_lower, {}):
        raise ValueError(f"Method '{method}' is not valid for tool '{tool}'.")

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['pet_file', 'anat_seg', 'anat_file', 'reg_file']),
        name='inputnode'
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['pet_pvc_file']), name='outputnode')

    pvc_params = pvc_params or {}

    method_config = config[tool_lower][method_key].copy()
    method_config.update(pvc_params)

    if tool_lower == 'petpvc':
        pvc_node = pe.Node(
            PETPVC(
                pvc=method_config.pop('pvc'),
                **method_config
            ),
            name=f'{tool_lower}_pvc_node',
        )

        workflow.connect([
            (inputnode, pvc_node, [('pet_file', 'in_file'), ('anat_seg', 'mask_file')]),
            (pvc_node, outputnode, [('out_file', 'pet_pvc_file')]),
        ])

    elif tool_lower == 'petsurfer':
        pvc_node = pe.Node(
            GTMPVC(**method_config),
            name=f'{tool_lower}_pvc_node',
        )

        workflow.connect([
            (inputnode, pvc_node, [
                ('pet_file', 'in_file'),
                ('anat_seg', 'segmentation'),
                ('reg_file', 'reg_file')
            ]),
            (pvc_node, outputnode, [('gtm_file', 'pet_pvc_file')]),
        ])

    else:
        raise ValueError(f"Unsupported PVC tool: {tool}")

    return workflow
