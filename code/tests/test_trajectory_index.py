import numpy as np

from dar_td3bc.data.trajectory_index import TrajectoryIndex


def test_future_flow_targets_do_not_cross_terminal_boundary():
    terminals = np.array(
        [False, False, False, False, True, False, False, False, False, True]
    )
    flow = np.arange(10, dtype=float)
    index = TrajectoryIndex(terminals)

    targets = index.future_flow_targets(flow, horizons=(1, 2, 5))

    assert targets.values.shape == (10, 3)
    assert targets.mask.shape == (10, 3)
    assert targets.horizons == (1, 2, 5)
    assert targets.values[3, 0] == 4.0
    assert targets.mask[3, 0] == 1.0
    assert targets.mask[3, 1] == 0.0
    assert targets.values[3, 1] == 0.0
    assert targets.mask[4, 0] == 0.0
    assert targets.values[5, 2] == 0.0
    assert targets.mask[5, 2] == 0.0


def test_future_flow_targets_respects_timeouts():
    terminals = np.array([False, False, False, False])
    timeouts = np.array([False, True, False, True])
    flow = np.array([10.0, 11.0, 20.0, 21.0])
    index = TrajectoryIndex(terminals, timeouts=timeouts)

    targets = index.future_flow_targets(flow, horizons=(1,))

    assert targets.values[:, 0].tolist() == [11.0, 0.0, 21.0, 0.0]
    assert targets.mask[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
