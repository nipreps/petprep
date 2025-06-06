import nibabel as nb
import numpy as np

from ..reference import init_raw_petref_wf


def test_reference_frame_select(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 4)), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    wf = init_raw_petref_wf(pet_file=str(pet_file), reference_frame=2)
    node_names = [n.name for n in wf._get_all_nodes()]
    assert 'extract_frame' in node_names
    assert 'gen_avg' not in node_names
    node = wf.get_node('extract_frame')
    assert node.interface.inputs.t_min == 2


def test_reference_frame_average(tmp_path):
    img = nb.Nifti1Image(np.zeros((5, 5, 5, 4)), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    img.to_filename(pet_file)

    wf = init_raw_petref_wf(pet_file=str(pet_file), reference_frame='average')
    node_names = [n.name for n in wf._get_all_nodes()]
    assert 'gen_avg' in node_names
