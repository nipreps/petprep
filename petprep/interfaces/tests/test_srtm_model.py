import numpy as np
from nipype.pipeline import Node, Workflow
from petprep.interfaces.kinmod import SRTM

def test_srtm_model():
    frame_times = np.array([10, 30, 50, 90, 150, 210, 300, 420, 540, 750, 1050, 1500, 
                        2100, 2700, 3300, 3900, 4500, 5100, 5700, 6300, 6900])
    reference_tac = np.array([-1.3000e+01,  1.3400e+02,  8.7000e+02,  3.0640e+03,  5.4180e+03,
        6.8900e+03,  9.7530e+03,  1.3040e+04,  1.5895e+04,  1.9864e+04,
        2.4001e+04,  2.6133e+04,  2.5346e+04,  2.2518e+04,  2.0362e+04,
        1.7967e+04,  1.5955e+04,  1.4412e+04,  1.3068e+04,  1.1734e+04,
        1.0224e+04])
    target_tac = np.array([-4.0000e+00, -5.0000e+00,  1.2040e+03,  3.9040e+03,  6.4980e+03,
        8.5220e+03,  1.1925e+04,  1.7069e+04,  2.1018e+04,  2.6893e+04,
        3.3922e+04,  4.0015e+04,  4.3574e+04,  4.3859e+04,  4.2365e+04,
        4.0321e+04,  3.7689e+04,  3.5427e+04,  3.1781e+04,  2.9542e+04,
        2.7551e+04])
    frame_weight = np.ones(len(target_tac))
    iterations = 10

    srtm_node = Node(SRTM(), name="srtm_model")
    srtm_node.inputs.reference_tac = reference_tac
    srtm_node.inputs.target_tac = target_tac
    srtm_node.inputs.frame_times = frame_times
    srtm_node.inputs.frame_weight = frame_weight
    srtm_node.inputs.iterations = iterations

    wf = Workflow(name="srtm_wf")
    wf.add_nodes([srtm_node])

    execgraph = wf.run()

    results = None
    for node in execgraph.nodes():
        if node.name == "srtm_model":
            results = node.result.outputs
            break

    assert results.R1 is not None
    assert results.k2 is not None
    assert results.BPnd is not None
    assert results.MSE is not None
    assert results.FPE is not None
    assert results.SigmaSqr is not None
    assert results.LogLike is not None
    assert results.AIC is not None

    assert isinstance(results.R1, float)
    assert isinstance(results.k2, float)
    assert isinstance(results.BPnd, float)
    assert isinstance(results.MSE, float)
    assert isinstance(results.FPE, float)
    assert isinstance(results.SigmaSqr, float)
    assert isinstance(results.LogLike, float)
    assert isinstance(results.AIC, float)
