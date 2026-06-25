import numpy as np

from dar_td3bc.envs.robustness_wrappers import (
    ObservationDelayWrapper,
    ObservationNoiseWrapper,
)


class CountingEnv:
    def __init__(self) -> None:
        self.value = 0

    def reset(self, *, seed=None, options=None):
        self.value = 0
        return np.array([0.0, 1.0], dtype=np.float32), {}

    def step(self, action):
        self.value += 1
        return (
            np.array([float(self.value), 1.0], dtype=np.float32),
            0.0,
            self.value >= 3,
            False,
            {},
        )


def test_observation_delay_wrapper_returns_delayed_observation():
    env = ObservationDelayWrapper(CountingEnv(), delay_steps=2)

    obs, _ = env.reset(seed=0)
    obs1, *_ = env.step(np.array([0.0]))
    obs2, *_ = env.step(np.array([0.0]))
    obs3, *_ = env.step(np.array([0.0]))

    assert obs.tolist() == [0.0, 1.0]
    assert obs1.tolist() == [0.0, 1.0]
    assert obs2.tolist() == [0.0, 1.0]
    assert obs3.tolist() == [1.0, 1.0]


def test_observation_noise_wrapper_with_zero_std_preserves_observation():
    env = ObservationNoiseWrapper(CountingEnv(), std=0.0)

    obs, _ = env.reset(seed=0)
    next_obs, *_ = env.step(np.array([0.0]))

    assert obs.tolist() == [0.0, 1.0]
    assert next_obs.tolist() == [1.0, 1.0]
