from __future__ import annotations

from nipype.interfaces import utility as niu
from nipype.interfaces.utility import Function
from nipype.pipeline import engine as pe

from ...interfaces import ExtractTACs


def resample_pet_to_segmentation(pet_file, segmentation_file):
    """Resample the PET image to the segmentation space."""

    import os

    from nilearn.image import resample_to_img

    resampled_pet = resample_to_img(pet_file, segmentation_file, interpolation='continuous')
    out_file = os.path.abspath('pet_resampled.nii.gz')

    resampled_pet.to_filename(out_file)
    return out_file


def init_pet_tacs_wf(
    *, name: str = 'pet_tacs_wf', resample_to_segmentation: bool = True
) -> pe.Workflow:
    """Extract time activity curves from a segmentation."""

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['pet_anat', 'pet_file', 'segmentation', 'dseg_tsv', 'metadata', 'support']
        ),
        name='inputnode',
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=['timeseries']), name='outputnode')

    # Resample PET to segmentation space
    resample_pet = pe.Node(
        Function(
            input_names=['pet_file', 'segmentation_file'],
            output_names=['resampled_pet'],
            function=resample_pet_to_segmentation,
        ),
        name='resample_pet',
    )

    tac = pe.Node(
        ExtractTACs(),
        name='tac',
    )

    if not resample_to_segmentation:
        workflow.connect(
            [
                (
                    inputnode,
                    tac,
                    [
                        ('pet_file', 'in_file'),
                        ('segmentation', 'segmentation'),
                        ('dseg_tsv', 'dseg_tsv'),
                        ('metadata', 'metadata'),
                        ('support', 'support'),
                    ],
                ),
                (tac, outputnode, [('out_file', 'timeseries')]),
            ]
        )
        return workflow

    workflow.connect(
        [
            (
                inputnode,
                resample_pet,
                [('pet_anat', 'pet_file'), ('segmentation', 'segmentation_file')],
            ),
            (
                resample_pet,
                tac,
                [('resampled_pet', 'in_file')],
            ),
            (
                inputnode,
                tac,
                [
                    ('segmentation', 'segmentation'),
                    ('dseg_tsv', 'dseg_tsv'),
                    ('metadata', 'metadata'),
                ],
            ),
            (tac, outputnode, [('out_file', 'timeseries')]),
        ]
    )

    return workflow


__all__ = ('init_pet_tacs_wf',)
