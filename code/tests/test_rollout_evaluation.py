from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dar_td3bc.evaluation.neorl_score import normalized_score
from dar_td3bc.evaluation.rollout import evaluate_checkpoint_rollout
from dar_td3bc.models.policies import BehaviorPolicy


class TinyPipelineEnv:
    def __init__(self) -> None:
        self.step_count = 0
        self.actions: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.step_count = 0
        self.actions = []
        return self._obs(flow=0.0, target=1.0), {}

    def step(self, action):
        scalar_action = float(np.asarray(action).reshape(-1)[0])
        self.actions.append(scalar_action)
        self.step_count += 1
        flow = float(self.step_count)
        target = 1.0
        reward = 1.0 - abs(flow - target)
        terminated = self.step_count >= 3
        truncated = False
        return self._obs(flow=flow, target=target), reward, terminated, truncated, {}

    def get_normalized_score(self, raw_return):
        return float(raw_return) * 10.0

    @staticmethod
    def _obs(*, flow: float, target: float) -> np.ndarray:
        obs = np.zeros(52, dtype=np.float32)
        obs[0] = flow
        obs[1] = target
        return obs


def _td3bc_checkpoint(path: Path) -> Path:
    actor = BehaviorPolicy(input_dim=52, hidden_dim=4)
    for param in actor.parameters():
        param.data.zero_()
    torch.save(
        {
            "config": {"model": {"hidden_dim": 4}},
            "actor": actor.state_dict(),
            "seed": 3,
        },
        path,
    )
    return path


def test_normalized_score_uses_environment_api():
    env = TinyPipelineEnv()

    assert normalized_score(env, 2.5) == 25.0


def test_evaluate_checkpoint_rollout_writes_neorl_metrics(tmp_path):
    checkpoint = _td3bc_checkpoint(tmp_path / "checkpoint.pt")

    output_path = evaluate_checkpoint_rollout(
        checkpoint_path=checkpoint,
        env_factory=TinyPipelineEnv,
        output_dir=tmp_path,
        method="td3bc_mlp",
        task="Pipeline",
        seeds=[0, 1],
        episodes_per_seed=2,
        max_steps=5,
    )

    frame = pd.read_csv(output_path)
    assert set(frame["provenance"]) == {"rollout_neorl2"}
    assert {"raw_return", "normalized_score", "tracking_rmse", "action_energy"} <= set(
        frame["metric"]
    )
    assert set(frame["seed"]) == {3}
    assert set(frame["eval_seed"]) == {0, 1}
    assert set(frame["episode"]) == {0, 1}
    assert np.isfinite(frame["value"]).all()
