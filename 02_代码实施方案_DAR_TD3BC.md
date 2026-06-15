# DAR-TD3BC 代码实现、数据准备与复现实验执行方案

> 本文件面向负责搭建工程、获取数据、实现算法、复现实验并生成论文结果的 AI Agent。Agent 应直接执行可完成的步骤；只有在数据或权限确实缺失时，才向用户提出一次明确的数据请求。

## 0. 最终目标

在 NeoRL-2 `Pipeline` 任务上实现并跑通：

**DAR-TD3BC = Delay-Aware Residual TD3+BC**

最终交付：

1. 官方数据可重复加载；
2. 普通 TD3+BC 内部基线可运行；
3. DAR-TD3BC 可训练和评估；
4. 消融、鲁棒性和统计实验可批量执行；
5. 结果自动汇总为 CSV、LaTeX 表格和论文图；
6. 官方文献结果与本地结果严格区分；
7. 所有代码、数据版本和随机种子可追溯。

---

## 1. 官方资源

### 1.1 NeoRL-2

- 官方环境仓库  
  https://github.com/polixir/NeoRL2

- 官方 Hugging Face 数据  
  https://huggingface.co/datasets/polixirai/NeoRL2

- 论文  
  https://arxiv.org/abs/2503.19267

官方基本调用：

```python
import neorl2
import gymnasium as gym

env = gym.make("Pipeline")
train_data, val_data = env.get_dataset()
```

预期字段应至少包含：

```text
observations
actions
rewards
next_observations
terminals / dones
```

实际字段名以当前版本为准，必须在预处理脚本中做映射并记录。

### 1.2 可复用实现

优先复用而不是手写未经验证的基础算法：

- TD3+BC 官方实现  
  https://github.com/sfujim/TD3_BC

- Polixir OfflineRL  
  https://github.com/polixir/OfflineRL

- CORL  
  https://github.com/tinkoff-ai/CORL

使用原则：

- TD3+BC 的 target smoothing、delayed policy update、normalization 和 actor scaling 参考官方实现；
- NeoRL-2 的环境、reward、score normalization 和数据接口使用官方代码；
- 本项目自己实现时序编码器、预测辅助任务、行为先验残差 actor、不确定性 gate 和控制评价；
- 第三方代码必须保留许可证和来源说明；
- 不直接复制 incompatible license 的代码；
- 记录所有第三方仓库 commit SHA。

---

## 2. 工作区和仓库组织

假设当前项目根目录为：

```text
workspace/dar_td3bc/
```

所有外部仓库克隆到项目根目录的并行目录：

```text
workspace/
├── dar_td3bc/        # 当前项目
├── NeoRL2/           # 官方环境与数据接口
├── OfflineRL/        # 官方 NeoRL-2 基线框架
├── TD3_BC/           # 原始 TD3+BC
└── CORL/             # 可选统一离线 RL 实现
```

克隆规则：

```bash
cd /path/to/workspace

test -d NeoRL2   || git clone https://github.com/polixir/NeoRL2.git NeoRL2
test -d OfflineRL || git clone https://github.com/polixir/OfflineRL.git OfflineRL
test -d TD3_BC   || git clone https://github.com/sfujim/TD3_BC.git TD3_BC
test -d CORL     || git clone https://github.com/tinkoff-ai/CORL.git CORL
```

不得把第三方仓库克隆到本项目内部。已有目录时不得覆盖未提交修改。

记录版本：

```bash
for d in NeoRL2 OfflineRL TD3_BC CORL; do
  git -C "../$d" rev-parse HEAD
done
```

写入：

```text
third_party_manifest.json
```

字段至少包括：

```json
{
  "name": "NeoRL2",
  "url": "https://github.com/polixir/NeoRL2",
  "commit": "...",
  "license": "...",
  "local_path": "../NeoRL2",
  "patch": null
}
```

如需修改第三方代码，创建 patch：

```bash
git -C ../NeoRL2 diff > patches/neorl2_pipeline.patch
```

---

## 3. 本项目目录结构

```text
dar_td3bc/
├── README.md
├── pyproject.toml
├── requirements.txt
├── third_party_manifest.json
├── configs/
│   ├── dataset/
│   │   └── pipeline.yaml
│   ├── model/
│   │   ├── td3bc_mlp.yaml
│   │   └── dar_td3bc.yaml
│   ├── train/
│   │   └── default.yaml
│   └── eval/
│       ├── default.yaml
│       └── robustness.yaml
├── data/
│   ├── raw/
│   │   └── neorl2/
│   ├── processed/
│   │   └── pipeline/
│   └── metadata/
├── src/
│   └── dar_td3bc/
│       ├── __init__.py
│       ├── data/
│       │   ├── pipeline_dataset.py
│       │   ├── trajectory_index.py
│       │   └── normalization.py
│       ├── envs/
│       │   ├── pipeline_factory.py
│       │   ├── delay_wrapper.py
│       │   └── noise_wrapper.py
│       ├── models/
│       │   ├── temporal_encoder.py
│       │   ├── behavior_policy.py
│       │   ├── residual_actor.py
│       │   ├── critic.py
│       │   └── prediction_head.py
│       ├── algorithms/
│       │   ├── td3bc.py
│       │   └── dar_td3bc.py
│       ├── evaluation/
│       │   ├── rollout.py
│       │   ├── control_metrics.py
│       │   ├── neorl_score.py
│       │   └── statistics.py
│       └── utils/
│           ├── seed.py
│           ├── checkpoint.py
│           ├── logging.py
│           └── provenance.py
├── scripts/
│   ├── inspect_environment.py
│   ├── acquire_data.py
│   ├── prepare_pipeline_data.py
│   ├── train_behavior_policy.py
│   ├── pretrain_encoder.py
│   ├── train_td3bc.py
│   ├── train_dar_td3bc.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   ├── run_robustness.py
│   ├── import_reported_baselines.py
│   ├── aggregate_results.py
│   └── make_paper_figures.py
├── tests/
│   ├── test_dataset.py
│   ├── test_obs_parser.py
│   ├── test_losses.py
│   ├── test_env_rollout.py
│   └── test_reproducibility.py
├── results/
│   ├── reported/
│   ├── runs/
│   ├── aggregated/
│   ├── tables/
│   └── figures/
├── patches/
└── paper/
```

---

## 4. 环境配置

### 4.1 Python 环境

优先使用 Python 3.10。Agent 应先检查官方依赖，不得盲目使用最新版本。

示例：

```bash
conda create -n dar-td3bc python=3.10 -y
conda activate dar-td3bc

python -m pip install --upgrade pip
pip install torch numpy scipy pandas matplotlib pyyaml tqdm \
  gymnasium datasets tensorboard pytest scikit-learn
pip install -e ../NeoRL2
pip install -e .
```

若 NeoRL-2 与 Gym/Gymnasium 版本冲突：

1. 读取 `../NeoRL2/setup.py`；
2. 建立锁定版本；
3. 记录在 `requirements-lock.txt`；
4. 不用 monkey patch 掩盖接口错误。

### 4.2 运行环境检查

`scripts/inspect_environment.py` 输出：

```text
Python version
OS
CPU
GPU and CUDA
RAM
free disk space
PyTorch version
Gymnasium version
NeoRL2 commit
network access
Hugging Face cache
```

结果保存到：

```text
results/environment_report.json
```

---

## 5. 数据获取与用户数据请求

### 5.1 自动获取顺序

Agent 按以下顺序执行，不得一开始就要求用户手动下载。

#### 方法 A：官方环境接口

```python
import neorl2
import gymnasium as gym

env = gym.make("Pipeline")
train_data, val_data = env.get_dataset()
```

#### 方法 B：Hugging Face

```python
from datasets import load_dataset

train = load_dataset(
    "polixirai/NeoRL2",
    "Pipeline",
    split="train"
)
```

实际配置名可能是 `polixir/neorl2` 或 `polixirai/NeoRL2`，以数据卡当前名称为准。

#### 方法 C：已有本地缓存

检查：

```text
data/raw/neorl2/
~/.cache/huggingface/datasets/
../NeoRL2/
```

### 5.2 确实缺少数据时向用户发送的唯一请求

Agent 完成代码框架和下载尝试后，如仍无法获得数据，向用户发送：

```text
当前代码和环境已准备完成，但 NeoRL-2 Pipeline 数据未能从官方接口下载。
请提供以下任一种：

A. Pipeline 训练集和验证集文件（Parquet、NPZ、PKL 均可）；
B. 已下载的 Hugging Face NeoRL-2 缓存目录；
C. 可联网下载的运行环境或代理方式；
D. 你本地执行 env.get_dataset() 后导出的数据目录。

数据至少需要以下字段：
observations、actions、rewards、next_observations、terminals/dones。

预期维度：
observations: [N, 52]
actions: [N, 1]
rewards: [N] 或 [N, 1]
next_observations: [N, 52]
terminals/dones: [N] 或 [N, 1]
```

不得只说“请提供数据”，必须给出上述字段和格式。

### 5.3 数据验证

`prepare_pipeline_data.py` 必须检查：

- 训练和验证样本数；
- 观测维数是否为 52；
- 动作维数是否为 1；
- 动作是否在 \([-1,1]\)；
- NaN/Inf；
- reward 范围；
- `next_observation` 连续性；
- episode 边界；
- 目标流量的取值；
- 历史队列顺序；
- train/val 是否重复。

输出：

```text
data/metadata/pipeline_manifest.json
data/metadata/pipeline_statistics.json
data/metadata/pipeline_validation_report.md
```

必须计算 SHA256。

---

## 6. Pipeline 环境源码核查

Agent 必须阅读当前 NeoRL-2 commit 的 Pipeline 环境源码和 reward 源码，不依赖二手描述。

当前已知重点：

- 源文件名可能拼写为 `neorl2/envs/pipline.py`；
- observation 前两维为当前下游流量和目标流量；
- 后续为 25 个历史流量和 25 个历史动作；
- 历史队列可能按“最近到最远”存储；
- action 为水门流量增量，环境内部水门流量被限制在 \([0,200]\)；
- agent action 被限制在 \([-1,1]\)；
- 默认 `channel_length=100`、`pipe_velocity=5`；
- 仿真运输队列长度为 `ceil(channel_length/pipe_velocity)=20`；
- 数据与论文通常把该任务概括为 25 步历史/延迟任务；
- target 可能以概率 0.003 在 \(\{50,80,110,140\}\) 中切换；
- episode horizon 通常为 1000；
- reward 只主要反映跟踪误差，动作平滑惩罚在源码中可能被注释。

Agent 将核查结果写入：

```text
docs/pipeline_source_audit.md
```

并附文件路径、行号和 commit。

---

## 7. 观测解析

建议实现：

```python
from dataclasses import dataclass
import torch


@dataclass
class ParsedPipelineObs:
    current_flow: torch.Tensor
    target_flow: torch.Tensor
    sequence: torch.Tensor
    flat: torch.Tensor


def split_pipeline_obs(obs: torch.Tensor) -> ParsedPipelineObs:
    if obs.ndim != 2 or obs.shape[-1] != 52:
        raise ValueError(f"Expected [B, 52], got {tuple(obs.shape)}")

    current_flow = obs[:, 0:1]
    target_flow = obs[:, 1:2]

    # 以当前官方源码为准；通常索引 2:27 和 27:52。
    flow_hist_recent_first = obs[:, 2:27]
    action_hist_recent_first = obs[:, 27:52]

    # 转成 oldest -> newest，供 GRU/TCN 使用。
    flow_hist = torch.flip(flow_hist_recent_first, dims=[1])
    action_hist = torch.flip(action_hist_recent_first, dims=[1])

    sequence = torch.stack([flow_hist, action_hist], dim=-1)  # [B, 25, 2]

    return ParsedPipelineObs(
        current_flow=current_flow,
        target_flow=target_flow,
        sequence=sequence,
        flat=obs,
    )
```

必须用人工构造序列测试历史顺序，不能只靠维数测试。

---

## 8. 轨迹与多步预测样本

数据通常以 transition 形式提供，而预测头需要未来流量。实现 `TrajectoryIndex`：

```python
class TrajectoryIndex:
    def __init__(self, terminals, timeouts=None, horizon=1000):
        ...
```

边界识别优先级：

1. 官方 `terminals`；
2. `timeouts/truncated`；
3. 数据自带 trajectory id；
4. 只有在验证连续性后才使用固定 1000 步切分。

多步目标：

```python
HORIZONS = (1, 5, 10, 20)
```

任一 horizon 跨 episode 时：

- 该 horizon mask 为 0；
- 不复制最后状态；
- 不跨轨迹索引。

---

## 9. 模型实现

### 9.1 时序编码器

首选轻量 TCN：

```python
import torch
from torch import nn


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.left_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y[..., :-self.left_padding] if self.left_padding > 0 else y


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size, dilation
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size, dilation
        )
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = y + residual
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class DelayEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Conv1d(2, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalConvBlock(hidden_dim, dilation=1),
            CausalConvBlock(hidden_dim, dilation=2),
            CausalConvBlock(hidden_dim, dilation=4),
            CausalConvBlock(hidden_dim, dilation=8),
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        current_flow: torch.Tensor,
        target_flow: torch.Tensor,
    ) -> torch.Tensor:
        # sequence: [B, 25, 2], oldest -> newest
        x = self.input_proj(sequence.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        h = x[:, :, -1]
        return self.head(torch.cat([h, current_flow, target_flow], dim=-1))
```

实现后必须检查因果卷积裁剪是否使输入输出长度一致。也可使用 GRU 作为第一版，之后再增加 TCN。

### 9.2 行为策略

```python
class BehaviorPolicy(nn.Module):
    def __init__(self, input_dim: int = 52, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
```

训练目标：

```python
loss_bc = ((behavior_policy(obs) - action) ** 2).mean()
```

使用验证集 early stopping，保存最佳 checkpoint。训练完整 RL 时默认冻结行为策略；另做可选联合微调消融。

### 9.3 残差 actor

```python
class ResidualActor(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)
```

最终动作：

```python
residual = residual_scale * actor(z)
action = torch.clamp(behavior_action + gate * residual, -1.0, 1.0)
```

`residual_scale` 初始建议搜索：

```text
0.1, 0.25, 0.5, 1.0
```

不得用测试集选择。

### 9.4 双 critic

```python
class TwinCritic(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.q1 = self._make_q(latent_dim, hidden_dim)
        self.q2 = self._make_q(latent_dim, hidden_dim)

    @staticmethod
    def _make_q(latent_dim: int, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        x = torch.cat([z, action], dim=-1)
        return self.q1(x), self.q2(x)
```

### 9.5 不确定性门控

基础形式：

```python
with torch.no_grad():
    q1, q2 = critic(z, candidate_action)
    disagreement = (q1 - q2).abs()
    gate = torch.exp(-kappa * normalized_disagreement)
    gate = gate.clamp(gate_min, 1.0)
```

注意：

- 若用 candidate action 计算 gate，需先定义未门控候选动作；
- gate 默认 `detach`；
- disagreement 需用训练分位数或运行均值归一化；
- 记录 gate 分布；
- 不能直接把双 critic 分歧称为严格 epistemic uncertainty。

### 9.6 多步预测头

```python
class MultiHorizonPredictionHead(nn.Module):
    def __init__(self, latent_dim: int, horizons=(1, 5, 10, 20)):
        super().__init__()
        self.horizons = tuple(horizons)
        self.head = nn.Linear(latent_dim, len(self.horizons))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
```

损失：

```python
sq_error = (pred - future_flow) ** 2
loss_pred = (sq_error * future_mask).sum() / future_mask.sum().clamp_min(1.0)
```

---

## 10. TD3+BC 与 DAR-TD3BC 更新逻辑

### 10.1 TD3 critic target

```python
with torch.no_grad():
    next_z = target_encoder(next_obs)
    next_behavior = behavior_policy(next_obs_flat)

    noise = (
        torch.randn_like(action) * policy_noise
    ).clamp(-noise_clip, noise_clip)

    next_residual = target_actor(next_z)
    next_action = torch.clamp(
        next_behavior + next_gate * next_residual + noise,
        -1.0, 1.0
    )

    target_q1, target_q2 = target_critic(next_z, next_action)
    target_q = reward + discount * (1.0 - done) * torch.minimum(
        target_q1, target_q2
    )
```

### 10.2 critic loss

```python
q1, q2 = critic(z, action)
critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
```

### 10.3 actor loss

```python
behavior_action = behavior_policy(obs_flat).detach()
residual = residual_scale * actor(z)
policy_action = torch.clamp(
    behavior_action + gate * residual,
    -1.0, 1.0
)

q1_pi, q2_pi = critic(z, policy_action)
q_pi = torch.minimum(q1_pi, q2_pi)

lambda_q = alpha / q_pi.abs().mean().detach().clamp_min(1e-6)

loss_q = -lambda_q * q_pi.mean()
loss_bc = F.mse_loss(policy_action, dataset_action)
loss_res = residual.pow(2).mean()
loss_tv = (policy_action - previous_dataset_action).pow(2).mean()

actor_loss = (
    loss_q
    + alpha_bc * loss_bc
    + alpha_res * loss_res
    + alpha_tv * loss_tv
)
```

### 10.4 编码器更新策略

第一版采用：

1. 先独立预训练 encoder + prediction head；
2. RL 阶段允许 encoder 由 critic loss 和 prediction loss 更新；
3. actor 更新时可选择 detach encoder，避免策略梯度破坏表征；
4. 做以下消融：
   - frozen encoder；
   - joint critic update；
   - full joint update。

### 10.5 目标网络

使用 Polyak update：

```python
@torch.no_grad()
def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for src, tgt in zip(source.parameters(), target.parameters()):
        tgt.data.mul_(1.0 - tau).add_(tau * src.data)
```

actor 和 target actor 每 `policy_delay` 步更新一次。

---

## 11. 数据归一化

只使用训练集计算：

```python
obs_mean = train_obs.mean(axis=0)
obs_std = train_obs.std(axis=0).clip(min=1e-6)
```

保存：

```text
data/processed/pipeline/normalization.npz
```

注意：

- 环境评估时使用同一统计量；
- reward 是否缩放需参考 TD3+BC 官方实现和 NeoRL-2 基线；
- 动作不做 z-score，只维持 \([-1,1]\)；
- current flow、target 和历史序列必须与 flat observation 使用一致的标准化；
- future-flow prediction target 可单独标准化，评估时反变换。

---

## 12. 训练流程

### Stage 0：Smoke test

```bash
python scripts/inspect_environment.py
python scripts/acquire_data.py --task Pipeline
python scripts/prepare_pipeline_data.py
pytest -q
```

### Stage 1：行为策略

```bash
python scripts/train_behavior_policy.py \
  --config configs/model/dar_td3bc.yaml \
  --seed 0
```

验收：

- train/val BC loss 收敛；
- 动作在 \([-1,1]\)；
- 与数据动作的 MAE 和 MSE 有记录；
- checkpoint 可加载。

### Stage 2：延迟编码器预训练

```bash
python scripts/pretrain_encoder.py \
  --config configs/model/dar_td3bc.yaml \
  --seed 0
```

验收：

- 1、5、10、20 步预测误差均有记录；
- 不跨 episode；
- 与“预测恒定当前流量”基线比较；
- latent 无 NaN。

### Stage 3：普通 TD3+BC 内部基线

```bash
python scripts/train_td3bc.py \
  --task Pipeline \
  --seed 0 \
  --steps 500000
```

先跑 1000 update 确认无错误，再跑完整训练。

### Stage 4：DAR-TD3BC

```bash
python scripts/train_dar_td3bc.py \
  --config configs/model/dar_td3bc.yaml \
  --seed 0
```

### Stage 5：多种子

```bash
for seed in 0 1 2 3 4; do
  python scripts/train_td3bc.py --task Pipeline --seed "$seed"
  python scripts/train_dar_td3bc.py \
    --config configs/model/dar_td3bc.yaml \
    --seed "$seed"
done
```

### Stage 6：消融与鲁棒性

```bash
python scripts/run_ablation.py --seeds 0 1 2 3 4
python scripts/run_robustness.py --seeds 0 1 2 3 4
```

---

## 13. 推荐初始超参数

以下只是起点，必须用 validation data 选择，不得看 test rollout 反复调参。

```yaml
seed: 0
device: cuda

dataset:
  task: Pipeline
  history_len: 25
  horizons: [1, 5, 10, 20]

model:
  latent_dim: 128
  hidden_dim: 256
  encoder: tcn
  tcn_channels: 128
  tcn_dilations: [1, 2, 4, 8]
  residual_scale: 0.25
  gate_min: 0.05
  gate_kappa: 1.0

train:
  batch_size: 256
  discount: 0.99
  tau: 0.005
  actor_lr: 0.0003
  critic_lr: 0.0003
  encoder_lr: 0.0003
  behavior_lr: 0.0003
  policy_noise: 0.2
  noise_clip: 0.5
  policy_delay: 2
  total_updates: 500000

loss:
  alpha_td3bc: 2.5
  alpha_bc: 1.0
  alpha_res: 0.01
  alpha_tv: 0.01
  alpha_pred: 1.0
```

搜索范围保持小而有依据：

```text
residual_scale: 0.1, 0.25, 0.5
gate_kappa: 0.25, 0.5, 1.0, 2.0
alpha_pred: 0.1, 0.5, 1.0
alpha_res: 0, 0.001, 0.01
alpha_tv: 0, 0.001, 0.01
```

---

## 14. 评估实现

### 14.1 Rollout

```python
@torch.no_grad()
def evaluate_policy(env, agent, episodes: int, seed_list: list[int]):
    records = []
    for seed in seed_list:
        obs, info = env.reset(seed=seed)
        done = False
        truncated = False
        episode = []
        while not (done or truncated):
            action, aux = agent.act(obs, deterministic=True)
            next_obs, reward, done, truncated, info = env.step(action)
            episode.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "gate": aux.get("gate"),
                "disagreement": aux.get("disagreement"),
            })
            obs = next_obs
        records.append(episode)
    return records
```

若环境本身不返回 termination，必须通过官方 TimeLimit 确认 1000 步截断。

### 14.2 控制指标

实现并单元测试：

```python
def rmse(y, r):
    return float(((y - r) ** 2).mean() ** 0.5)

def mae(y, r):
    return float(abs(y - r).mean())

def iae(y, r, dt=1.0):
    return float(abs(y - r).sum() * dt)

def action_total_variation(a):
    return float(abs(a[1:] - a[:-1]).sum())

def action_energy(a):
    return float((a ** 2).sum())
```

目标变化事件：

- 检测 `r[t] != r[t-1]`；
- 对每次变化计算最大超调；
- 调节带为 `max(0.02*abs(target), 2.0)`；
- 连续 10 步进入带内才认为 settled；
- episode 结束仍未稳定时记为 censored，并单独报告失败比例。

### 14.3 官方 normalized return

不得自行猜测公式。Agent 应：

1. 搜索 NeoRL-2 benchmark 中的 `report_result.py`、`task_score.csv` 或 score function；
2. 调用官方实现；
3. 在 `src/dar_td3bc/evaluation/neorl_score.py` 中封装；
4. 单元测试官方已知 behavior policy score；
5. 记录公式和版本。

---

## 15. 官方基线结果导入策略

用户允许在官方仓库或论文已有 Pipeline 结果时不重新运行。执行方式：

```text
results/reported/neorl2_pipeline_official.csv
```

字段：

```text
method
mean
error
error_type
metric
num_seeds
source_title
source_url
source_table
source_commit
retrieved_date
provenance
notes
```

`provenance` 固定为：

```text
reported_not_rerun
```

导入前必须从原始论文或官方仓库双重核对。

论文表格中必须写：

> Reported baseline values are reproduced from Gao et al. under their original implementation and are not re-run in our codebase.

不得：

- 把官方误差栏误写为本项目标准差；
- 将不同评估 episode 数的结果直接做显著性检验；
- 从图上目测抄数字；
- 复制第三方博客数字；
- 省略 provenance。

---

## 16. 没有现成 Pipeline 结果的基线

对于确实需要而官方没有结果的方法：

1. 在项目并行目录克隆官方仓库；
2. 记录 commit；
3. 创建适配脚本，不修改其核心算法；
4. 使用相同数据、归一化、seed 和评估环境；
5. 保存完整配置；
6. 如无法复现，记录失败原因，不虚构结果。

执行模板：

```bash
cd ..
git clone <OFFICIAL_REPO_URL> <METHOD_NAME>
cd dar_td3bc

python scripts/adapt_external_baseline.py \
  --repo ../<METHOD_NAME> \
  --task Pipeline

python scripts/run_external_baseline.py \
  --method <METHOD_NAME> \
  --seeds 0 1 2 3 4
```

输出：

```text
results/runs/external/<method>/<commit>/<seed>/
```

只复现与论文论证直接相关的方法。默认不需要复现所有 10 个 NeoRL-2 官方基线。

---

## 17. 消融配置

至少创建：

```text
td3bc_mlp
td3bc_tcn
td3bc_tcn_pred
residual_td3bc
residual_td3bc_gate
dar_td3bc_no_tv
dar_td3bc_full
```

固定：

- 数据；
- seed；
- batch；
-训练更新数；
- critic/actor 容量尽量相近；
- checkpoint 选择规则。

为避免“参数更多所以更好”，增加参数匹配的 MLP baseline。

---

## 18. 鲁棒性环境

### 18.1 噪声 wrapper

```python
class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, std, obs_mean, obs_std):
        super().__init__(env)
        self.std = std
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def observation(self, observation):
        normalized = (observation - self.obs_mean) / self.obs_std
        noise = self.np_random.normal(0.0, self.std, size=normalized.shape)
        return normalized + noise
```

注意不要重复归一化。更稳妥的方式是在 agent 输入端注入噪声。

### 18.2 延迟变化

不要覆盖官方环境文件。创建：

```text
src/dar_td3bc/envs/delay_wrapper.py
```

若运输队列不能通过 wrapper 合法调整，则复制最少环境逻辑到本项目并：

- 标明来源；
- 保留许可证；
- 在文件头记录原 commit；
- 只修改 `channel_length` 或 `pipe_velocity`；
- 写测试确认队列长度。

默认主结果只使用官方环境。

### 18.3 数据子采样

按完整 trajectory 抽样，不能逐 transition 随机抽样破坏时序。固定 trajectory id 和 seed，保存索引文件。

---

## 19. 结果与日志格式

每次运行目录：

```text
results/runs/<method>/<timestamp>_<seed>/
├── config_resolved.yaml
├── environment.json
├── git_state.json
├── metrics_train.csv
├── metrics_validation.csv
├── metrics_eval.csv
├── trajectories.npz
├── checkpoint_best.pt
├── checkpoint_last.pt
└── stdout.log
```

`git_state.json` 包含：

- 当前项目 commit；
- dirty status；
- NeoRL-2 commit；
- 数据 hash；
- 命令行；
- seed；
- hostname；
- CUDA/PyTorch。

---

## 20. 聚合与统计

`aggregate_results.py`：

1. 扫描所有运行；
2. 排除失败运行时给出原因；
3. 按方法和设置聚合；
4. 输出 mean、std、median、95% CI；
5. 执行配对检验；
6. 输出 effect size；
7. 生成 `final_summary.csv`。

输出：

```text
results/aggregated/final_summary.csv
results/aggregated/paired_tests.csv
results/tables/main_results.tex
results/tables/ablation.tex
results/tables/robustness.tex
results/figures/*.pdf
results/figures/*.png
```

图使用矢量 PDF；预览 PNG 至少 300 dpi。论文数值只能从这些文件自动读取。

---

## 21. 测试要求

### 21.1 单元测试

至少覆盖：

- 52 维 observation 解析；
- 历史顺序；
- episode 边界；
- 多步预测 mask；
- normalization；
- action clipping；
- gate 范围；
- target network update；
- actor/critic loss finite；
- control metrics；
- score wrapper。

### 21.2 集成测试

顺序：

1. 环境随机动作 100 步；
2. 加载 1024 条数据；
3. BC 训练 100 step；
4. encoder 训练 100 step；
5. TD3BC 更新 100 step；
6. DAR-TD3BC 更新 100 step；
7. 评估一个 1000 步 episode；
8. checkpoint save/load 后动作一致；
9. CPU 和 GPU 均可运行；
10. 固定 seed 的短运行结果可重复。

### 21.3 失败处理

出现 NaN 时自动保存：

- batch；
- network norms；
- Q distribution；
- reward range；
- gradient norms；
- config；
- seed。

不得简单减小学习率后隐藏原因。

---

## 22. Agent 执行顺序

Agent 应按以下阶段工作并持续更新 `PROGRESS.md`：

### Phase A：核查与搭建

- 检查设备、网络和目录；
- 克隆/发现第三方仓库；
- 记录 commits；
- 创建环境；
- 运行 NeoRL-2 随机 rollout。

### Phase B：数据

- 自动下载；
- 验证字段、形状和轨迹；
- 生成 hash 和统计；
- 若失败，再向用户发出第 5.2 节的明确请求。

### Phase C：基础基线

- 导入官方报告基线；
- 实现并跑通内部 TD3+BC；
- 确认 normalized score 函数。

### Phase D：方法

- BC behavior prior；
- delay encoder；
- prediction head；
- residual actor；
- uncertainty gate；
- 完整训练。

### Phase E：实验

- 5 seeds；
- 消融；
- 鲁棒性；
- 统计；
- 图表。

### Phase F：交付

- 更新 README；
- 给出一条从零复现命令；
- 输出结果表和图；
- 给论文 Agent 提供 `results_manifest.json`；
- 列出未完成或失败实验。

---

## 23. 最低可运行版本与完整版本

### 最低可运行版本

资源不足时，先完成：

- 数据加载；
- 1 个 seed TD3+BC；
- 1 个 seed DAR-TD3BC；
- 20 个评估 episode；
- tracking 曲线；
- RMSE、return、action TV；
- README 复现命令。

### 论文完整版本

投稿前必须完成：

- 5 个训练 seed；
- 内部普通 TD3+BC；
- 完整方法；
- 至少 4 个关键消融；
- 至少 3 个鲁棒性维度；
- 统计检验；
- 参数量和推理延迟；
- 所有配置与数据 hash；
- 官方基线 provenance；
- 全部结果自动汇总。

---

## 24. 最终验收标准

代码部分只有满足以下条件才算完成：

- `pytest -q` 全部通过；
- Pipeline 数据与环境可加载；
- observation 解析与源码一致；
- 一条命令可训练；
- 一条命令可评估；
- 一条命令可生成论文图表；
- 至少一个 seed 从数据到结果完整跑通；
- 完整实验至少 5 seeds；
- 没有 NaN；
- 行为先验和完整策略动作均在 \([-1,1]\)；
- reported 与 rerun 结果有明确 provenance；
- 任何复制的官方数字都有原始链接、表号和检索日期；
- 任何外部方法复现都有 commit 和配置；
- 论文所用每个数字均能追溯到原始结果文件；
- README 包含环境、数据、训练、评估和复现说明。
