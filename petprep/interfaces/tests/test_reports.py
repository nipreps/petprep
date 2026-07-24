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

import pytest

from ..reports import get_world_pedir


@pytest.mark.parametrize(
    ('orientation', 'pe_dir', 'expected'),
    [
        ('RAS', 'j', 'Posterior-Anterior'),
        ('RAS', 'j-', 'Anterior-Posterior'),
        ('RAS', 'i', 'Left-Right'),
        ('RAS', 'i-', 'Right-Left'),
        ('RAS', 'k', 'Inferior-Superior'),
        ('RAS', 'k-', 'Superior-Inferior'),
        ('LAS', 'j', 'Posterior-Anterior'),
        ('LAS', 'i-', 'Left-Right'),
        ('LAS', 'k-', 'Superior-Inferior'),
        ('LPI', 'j', 'Anterior-Posterior'),
        ('LPI', 'i-', 'Left-Right'),
        ('LPI', 'k-', 'Inferior-Superior'),
        ('SLP', 'k-', 'Posterior-Anterior'),
        ('SLP', 'k', 'Anterior-Posterior'),
        ('SLP', 'j-', 'Left-Right'),
        ('SLP', 'j', 'Right-Left'),
        ('SLP', 'i', 'Inferior-Superior'),
        ('SLP', 'i-', 'Superior-Inferior'),
    ],
)
def test_get_world_pedir(tmpdir, orientation, pe_dir, expected):
    assert get_world_pedir(orientation, pe_dir) == expected


def test_subject_summary_handles_missing_task(tmp_path):
    from ..reports import SubjectSummary

    t1w = tmp_path / 'sub-01_T1w.nii.gz'
    t1w.write_text('')
    pet1 = tmp_path / 'sub-01_task-rest_run-01_pet.nii.gz'
    pet1.write_text('')
    pet2 = tmp_path / 'sub-01_run-01_pet.nii.gz'
    pet2.write_text('')

    summary = SubjectSummary(
        t1w=[str(t1w)],
        pet=[str(pet1), str(pet2)],
        std_spaces=[],
        nstd_spaces=[],
    )

    segment = summary._generate_segment()
    assert 'PET series: 2' in segment
    assert 'Task: rest (1 run)' in segment
    assert 'Task: <none> (1 run)' in segment


@pytest.mark.parametrize(
    'registration',
    ['mri_coreg', 'mri_robust_register', 'ants_registration'],
)
def test_functional_summary_with_metadata(registration):
    from ..reports import PETSummary

    summary = PETSummary(
        registration=registration,
        registration_dof=6,
        orientation='RAS',
        anatref_strategy='t1w',
        requested_anatref='auto',
        volume_ratio=1.6,
        reference_policy='t1w-pre-masked-cropped',
        petref_strategy='template',
        metadata={
            'TracerName': 'DASB',
            'TracerRadionuclide': '[11C]',
            'InjectedRadioactivity': 100,
            'InjectedRadioactivityUnits': 'MBq',
            'FrameTimesStart': [0, 1],
            'FrameDuration': [1, 1],
        },
    )

    segment = summary._generate_segment()
    assert registration in segment
    assert 'Reference image: Motion correction template' in segment
    assert (
        'Registration reference policy: Preprocessed T1w image (brain-masked, cropped)' in segment
    )
    assert (
        'Anatomical reference: Preprocessed T1w image '
        "(PET/T1w mask volume ratio: 1.60) (requested 'auto')" in segment
    )
    assert 'Radiotracer: [11C]DASB' in segment
    assert 'Injected dose: 100 MBq' in segment
    assert 'Number of frames: 2' in segment


@pytest.mark.parametrize('winner, expected', [('ants', 'ANTs'), ('freesurfer', 'FreeSurfer')])
def test_functional_summary_auto_select(winner, expected):
    from ..reports import PETSummary

    summary = PETSummary(
        registration='auto_select',
        registration_dof=6,
        orientation='RAS',
        anatref_strategy='t1w',
        petref_strategy='template',
        metadata={},
        registration_winner=winner,
    )

    segment = summary._generate_segment()
    assert f'Automatic selection between FreeSurfer and ANTs (best score: {expected})' in segment
    assert 'Registration reference policy:' not in segment


def test_functional_summary_auto_select_reports_similarity_metric():
    from ..reports import PETSummary

    summary = PETSummary(
        registration='auto_select',
        registration_dof=6,
        orientation='RAS',
        anatref_strategy='t1w',
        petref_strategy='template',
        metadata={},
        registration_winner='freesurfer',
        registration_score=-0.1,
    )

    segment = summary._generate_segment()
    assert (
        'Automatic selection between FreeSurfer and ANTs '
        '(best score: FreeSurfer; similarity metric: -0.1)' in segment
    )


def test_functional_summary_formats_nu_reference_policy():
    from ..reports import PETSummary

    summary = PETSummary(
        registration='mri_coreg',
        registration_dof=6,
        orientation='RAS',
        anatref_strategy='nu',
        reference_policy='nu-unmasked-cropped',
        petref_strategy='template',
        metadata={},
    )

    segment = summary._generate_segment()
    assert 'Registration reference policy: FreeSurfer nu.mgz (unmasked, cropped)' in segment


def test_atlas_rois_report(tmp_path):
    import nibabel as nb
    import numpy as np

    from ..reports import AtlasROIsReport

    affine = np.diag([2, 2, 2, 1])
    t1_data = np.zeros((12, 12, 12), dtype=np.float32)
    pet_data = np.zeros((12, 12, 12), dtype=np.float32)
    seg_data = np.zeros((12, 12, 12), dtype=np.uint8)
    seg_data[3:9, 3:9, 3:6] = 1
    seg_data[4:10, 4:10, 6:9] = 2

    t1_file = tmp_path / 't1.nii.gz'
    pet_file = tmp_path / 'pet.nii.gz'
    seg_file = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(t1_data, affine).to_filename(t1_file)
    nb.Nifti1Image(pet_data, affine).to_filename(pet_file)
    nb.Nifti1Image(seg_data, affine).to_filename(seg_file)

    tsv_file = tmp_path / 'atlas.tsv'
    tsv_file.write_text('index\tname\n1\tRegionA\n2\tRegionB\n')

    report = AtlasROIsReport(
        t1w_image=str(t1_file),
        petref_image=str(pet_file),
        segmentation=str(seg_file),
        dseg_tsv=str(tsv_file),
        atlas_name='TestAtlas',
    )
    result = report.run(cwd=tmp_path)
    # assert result.outputs.out_file
    # assert Path(result.outputs.out_file).exists()
