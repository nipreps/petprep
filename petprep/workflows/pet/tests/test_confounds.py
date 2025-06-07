import nibabel as nb
import numpy as np

from ..confounds import init_pet_confs_wf


def test_dvars_connects_pet_mask(tmp_path):
    """Check dvars node connection and execution."""
    wf = init_pet_confs_wf(
        mem_gb=0.01,
        metadata={},
        regressors_all_comps=False,
        regressors_dvars_th=1.5,
        regressors_fd_th=0.5,
    )

    edge = wf._graph.get_edge_data(wf.get_node('inputnode'), wf.get_node('dvars'))
    assert ('pet_mask', 'in_mask') in edge['connect']

    img = nb.Nifti1Image(np.random.rand(2, 2, 2, 5), np.eye(4))
    mask = nb.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    pet_file = tmp_path / 'pet.nii.gz'
    mask_file = tmp_path / 'mask.nii.gz'
    img.to_filename(pet_file)
    mask.to_filename(mask_file)

    node = wf.get_node('dvars')
    node.base_dir = tmp_path
    node.inputs.in_file = str(pet_file)
    node.inputs.in_mask = str(mask_file)
    result = node.run()

    assert result.outputs.out_nstd
    assert result.outputs.out_std
