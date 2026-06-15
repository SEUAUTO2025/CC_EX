import math

from dar_td3bc.evaluation.control_metrics import (
    action_energy,
    action_total_variation,
    iae,
    mae,
    rmse,
    target_switch_settling_events,
)


def test_basic_tracking_and_action_metrics():
    y = [1.0, 3.0, 5.0]
    r = [1.0, 1.0, 2.0]
    action = [0.0, 0.5, -0.25]

    assert math.isclose(rmse(y, r), math.sqrt((0.0 + 4.0 + 9.0) / 3.0))
    assert math.isclose(mae(y, r), (0.0 + 2.0 + 3.0) / 3.0)
    assert math.isclose(iae(y, r, dt=0.5), (0.0 + 2.0 + 3.0) * 0.5)
    assert math.isclose(action_total_variation(action), 1.25)
    assert math.isclose(action_energy(action), 0.3125)


def test_target_switch_settling_reports_settled_and_censored_events():
    target = [50.0] * 3 + [80.0] * 12 + [110.0] * 4
    flow = [50.0] * 3 + [70.0, 76.0] + [80.5] * 10 + [90.0] * 4

    events = target_switch_settling_events(flow, target, hold_steps=3)

    assert len(events) == 2
    first = events[0]
    assert first.switch_index == 3
    assert first.previous_target == 50.0
    assert first.target == 80.0
    assert first.settled_index == 5
    assert first.settling_time == 2.0
    assert first.censored is False

    second = events[1]
    assert second.switch_index == 15
    assert second.target == 110.0
    assert second.settled_index is None
    assert second.settling_time is None
    assert second.censored is True
