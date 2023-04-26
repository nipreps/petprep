import numpy as np
from nipype.pipeline import Node, Workflow
from petprep.interfaces.kinmod import LoganModel

def test_logan_model():
    frame_times = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
    reference_tac = np.array([0, 10, 20, 25, 28, 30, 31, 31.5, 32, 32.2, 32.3])
    target_tac = np.array([0, 5, 12, 17, 20, 21, 22, 22.5, 23, 23.2, 23.3])
    tstar = 90
    k2ref = 0.05
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    logan_node = Node(LoganModel(), name="logan_model")
    logan_node.inputs.reference_tac = reference_tac
    logan_node.inputs.target_tac = target_tac
    logan_node.inputs.frame_times = frame_times
    logan_node.inputs.tstar = tstar
    logan_node.inputs.k2ref = k2ref
    logan_node.inputs.weights = weights

    wf = Workflow(name="logan_wf")
    wf.add_nodes([logan_node])

    execgraph = wf.run()

    results = None
    for node in execgraph.nodes():
        if node.name == "logan_model":
            results = node.result.outputs
            break

    assert results.Kappa2 is not None
    assert results.BPnd is not None
    assert results.MSE is not None
    assert results.FPE is not None
    assert results.SigmaSqr is not None
    assert results.LogLike is not None
    assert results.AIC is not None

    assert isinstance(results.Kappa2, float)
    assert isinstance(results.BPnd, float)
    assert isinstance(results.MSE, float)
    assert isinstance(results.FPE, float)
    assert isinstance(results.SigmaSqr, float)
    assert isinstance(results.LogLike, float)
    assert isinstance(results.AIC, float)
