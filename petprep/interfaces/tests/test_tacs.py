import gzip
import json
import pickle
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from nipype.pipeline import engine as pe
from nipype.pipeline.engine.nodes import NodeExecutionError

from petprep.interfaces.tacs import ExtractRefTAC, ExtractTACs
from petprep.workflows.pet.ref_tacs import init_pet_ref_tacs_wf
from petprep.workflows.pet.tacs import init_pet_tacs_wf


def test_ExtractTACs(tmp_path):
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    seg_data = np.tile([[1, 2], [1, 2]], (2, 1, 1)).astype('int16')
    seg_file = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(seg_data, np.eye(4)).to_filename(seg_file)

    dseg_tsv = tmp_path / 'seg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(dseg_tsv, sep='\t', index=False)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0, 1], 'FrameDuration': [1, 1]}))

    node = pe.Node(
        ExtractTACs(
            in_file=str(pet_file),
            segmentation=str(seg_file),
            dseg_tsv=str(dseg_tsv),
            metadata=str(meta_json),
        ),
        name='tac',
        base_dir=tmp_path,
    )
    res = node.run()

    out = pd.read_csv(res.outputs.out_file, sep='\t')
    assert list(out.columns) == ['frame_start', 'frame_end', 'A', 'B']
    assert np.allclose(out['A'], [1, 2])
    assert np.allclose(out['B'], [1, 2])


def test_ExtractTACs_mismatched_meta(tmp_path):
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    seg_data = np.tile([[1, 2], [1, 2]], (2, 1, 1)).astype('int16')
    seg_file = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(seg_data, np.eye(4)).to_filename(seg_file)

    dseg_tsv = tmp_path / 'seg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(dseg_tsv, sep='\t', index=False)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0], 'FrameDuration': [1, 1]}))

    node = pe.Node(
        ExtractTACs(
            in_file=str(pet_file),
            segmentation=str(seg_file),
            dseg_tsv=str(dseg_tsv),
            metadata=str(meta_json),
        ),
        name='tac',
        base_dir=tmp_path,
    )

    with pytest.raises(NodeExecutionError):
        node.run()


def test_ExtractTACs_mismatched_frames(tmp_path):
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    seg_data = np.tile([[1, 2], [1, 2]], (2, 1, 1)).astype('int16')
    seg_file = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(seg_data, np.eye(4)).to_filename(seg_file)

    dseg_tsv = tmp_path / 'seg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(dseg_tsv, sep='\t', index=False)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0, 1, 2], 'FrameDuration': [1, 1, 1]}))

    node = pe.Node(
        ExtractTACs(
            in_file=str(pet_file),
            segmentation=str(seg_file),
            dseg_tsv=str(dseg_tsv),
            metadata=str(meta_json),
        ),
        name='tac',
        base_dir=tmp_path,
    )

    with pytest.raises(NodeExecutionError):
        node.run()


def test_tacs_workflow(tmp_path):
    """Workflow passes resampled PET to ExtractTACs."""
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    seg_data = np.tile([[1, 2], [1, 2]], (2, 1, 1)).astype('int16')
    seg_file = tmp_path / 'seg.nii.gz'
    nb.Nifti1Image(seg_data, np.eye(4)).to_filename(seg_file)

    dseg_tsv = tmp_path / 'seg.tsv'
    pd.DataFrame({'index': [1, 2], 'name': ['A', 'B']}).to_csv(dseg_tsv, sep='\t', index=False)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0, 1], 'FrameDuration': [1, 1]}))

    wf = init_pet_tacs_wf()
    wf.base_dir = str(tmp_path)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.inputs.inputnode.pet_anat = str(pet_file)
    wf.inputs.inputnode.segmentation = str(seg_file)
    wf.inputs.inputnode.dseg_tsv = str(dseg_tsv)
    wf.inputs.inputnode.metadata = str(meta_json)

    wf.run()

    resampled_pet = tmp_path / 'pet_tacs_wf' / 'resample_pet' / 'pet_resampled.nii.gz'
    tac_inputs_file = tmp_path / 'pet_tacs_wf' / 'tac' / '_inputs.pklz'
    with gzip.open(tac_inputs_file, 'rb') as f:
        inputs = pickle.load(f)

    assert inputs['in_file'] == str(resampled_pet)
    assert Path(tmp_path / 'pet_tacs_wf' / 'tac' / 'pet_resampled_tacs.tsv').exists()


def test_ExtractRefTAC(tmp_path):
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    mask_data = np.zeros((2, 2, 2), dtype='int16')
    mask_data[0] = 1
    mask_file = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0, 1], 'FrameDuration': [1, 1]}))

    node = pe.Node(
        ExtractRefTAC(
            in_file=str(pet_file),
            mask_file=str(mask_file),
            ref_mask_name='ref',
            metadata=str(meta_json),
        ),
        name='tac',
        base_dir=tmp_path,
    )
    res = node.run()

    out = pd.read_csv(res.outputs.out_file, sep='\t')
    assert list(out.columns) == ['frame_start', 'frame_end', 'ref']
    assert np.allclose(out['ref'], [1, 2])


def test_ExtractRefTAC_mismatched_meta(tmp_path):
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    mask_data = np.zeros((2, 2, 2), dtype='int16')
    mask_data[0] = 1
    mask_file = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0], 'FrameDuration': [1, 1]}))

    node = pe.Node(
        ExtractRefTAC(
            in_file=str(pet_file),
            mask_file=str(mask_file),
            ref_mask_name='ref',
            metadata=str(meta_json),
        ),
        name='tac2',
        base_dir=tmp_path,
    )

    with pytest.raises(NodeExecutionError):
        node.run()


def test_ExtractRefTAC_mismatched_frames(tmp_path):
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    mask_data = np.zeros((2, 2, 2), dtype='int16')
    mask_data[0] = 1
    mask_file = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0, 1, 2], 'FrameDuration': [1, 1, 1]}))

    node = pe.Node(
        ExtractRefTAC(
            in_file=str(pet_file),
            mask_file=str(mask_file),
            ref_mask_name='ref',
            metadata=str(meta_json),
        ),
        name='tac2',
        base_dir=tmp_path,
    )

    with pytest.raises(NodeExecutionError):
        node.run()


def test_ref_tacs_workflow_mismatched_meta(tmp_path):
    """Workflow should fail with inconsistent metadata."""
    pet_data = np.stack(
        [
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)) * 2,
        ],
        axis=-1,
    )
    pet_file = tmp_path / 'pet.nii.gz'
    nb.Nifti1Image(pet_data, np.eye(4)).to_filename(pet_file)

    mask_data = np.zeros((2, 2, 2), dtype='int16')
    mask_data[0] = 1
    mask_file = tmp_path / 'mask.nii.gz'
    nb.Nifti1Image(mask_data, np.eye(4)).to_filename(mask_file)

    meta_json = tmp_path / 'pet.json'
    meta_json.write_text(json.dumps({'FrameTimesStart': [0], 'FrameDuration': [1, 1]}))

    wf = init_pet_ref_tacs_wf()
    wf.base_dir = str(tmp_path)
    wf.config['execution']['remove_unnecessary_outputs'] = False
    wf.inputs.inputnode.pet_anat = str(pet_file)
    wf.inputs.inputnode.mask_file = str(mask_file)
    wf.inputs.inputnode.metadata = str(meta_json)
    wf.inputs.inputnode.ref_mask_name = 'ref'

    with pytest.raises(NodeExecutionError):
        wf.run()
