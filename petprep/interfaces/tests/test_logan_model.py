import numpy as np
from nipype.pipeline import Node, Workflow
from petprep.interfaces.kinmod import LoganModel

def test_logan_model():
    frame_times = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
    reference_tac = np.array([0, 10, 20, 25, 28, 30, 31, 31.5, 32, 32.2, 32.3])
    target_tac = np.array([0, 5, 12, 17, 20, 21, 22, 22.5, 23, 23.2, 23.3])
    start_time = 90

    logan_node = Node(LoganModel(), name="logan_model")
    logan_node.inputs.reference_tac = reference_tac
    logan_node.inputs.target_tac = target_tac
    logan_node.inputs.frame_times = frame_times
    logan_node.inputs.start_time = start_time

    wf = Workflow(name="logan_wf")
    wf.add_nodes([logan_node])

    execgraph = wf.run()

    results = None
    for node in execgraph.nodes():
        if node.name == "logan_model":
            results = node.result.outputs
            break

    assert results.Kappa2 is not None
    assert results.VT is not None
    assert isinstance(results.Kappa2, float)
    assert isinstance(results.VT, float)
