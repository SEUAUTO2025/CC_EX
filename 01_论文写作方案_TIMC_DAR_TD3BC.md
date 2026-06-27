# DAR-TD3BC 论文写作与投稿执行方案

> 本文件面向能够联网检索、读取项目结果并生成英文论文初稿的 AI Agent。Agent 必须按本文执行，不得虚构实验、引用、期刊规则或结论。

## 0. 任务定义

围绕 NeoRL-2 的 `Pipeline` 延迟过程控制任务，研究一种面向离线数据的延迟感知残差强化学习方法：

**DAR-TD3BC: Delay-Aware Residual TD3+BC for Offline Control of Delayed Pipeline Systems**

核心方法由四部分组成：

1. 因果时序编码器：从流量与动作历史中提取延迟相关隐状态；
2. 多步流量预测辅助任务：显式强化动作—响应延迟表征；
3. 行为先验残差策略：策略只在行为策略附近输出受限修正量；
4. 基于双 Q 网络分歧的自适应残差门控：在价值估计不确定时自动减小策略偏移。

研究目标不是声称解决真实工业管道部署，而是在一个面向近真实工业控制问题的公开离线 RL 基准上，验证延迟感知、保守策略改进和控制平滑性设计的有效性。

---

## 1. 目标期刊

### 1.1 首选期刊

**Transactions of the Institute of Measurement and Control（TIMC）**  
Publisher: SAGE  
ISSN: 0142-3312；Online ISSN: 1477-0369

官方主页：

- https://journals.sagepub.com/home/tim
- https://journals.sagepub.com/author-instructions/tim

选择理由：

- 期刊范围覆盖 measurement、control、automation、system modelling、computational intelligence 和控制系统实现；
- 接收控制算法、智能控制、数据驱动控制和强化学习控制方向的 Original Article；
- 相比控制领域顶级期刊，对“方法创新 + 标准公开任务 + 系统控制指标”的完整应用型研究更匹配；
- 采用订阅发表时，官方当前说明不收投稿费和常规发表费；开放获取为可选；
- 官方当前显示 Impact Factor 1.9、5-year Impact Factor 1.7。

### 1.2 分区核验要求

“SCI 三区或四区”会随年份、学科类别和学校口径变化。Agent 在正式投稿前必须执行以下核验：

1. 在 Clarivate Master Journal List 确认期刊仍被 **SCIE** 收录；
2. 在最新 Journal Citation Reports 中检查：
   - `Automation & Control Systems`
   - `Instruments & Instrumentation`
   两个类别的 JCR Quartile；
3. 将核验日期、JCR 年份和分区记录在 `paper/journal_check.md`；
4. 若最新 JCR 已升至 Q2，先向用户报告，再由用户决定继续投稿或切换到当年明确的 Q3/Q4 期刊；
5. 不得用 SCImago 的 SJR Quartile 冒充 JCR Quartile。

检索入口：

- Clarivate Master Journal List: https://mjl.clarivate.com/
- Journal Citation Reports: https://jcr.clarivate.com/
- SAGE 官方投稿要求: https://journals.sagepub.com/author-instructions/tim

### 1.3 官方格式约束

以投稿时的官方说明为最终依据。当前方案按以下要求设计：

- Article type: **Original Article**
- 正文通常不超过 **6000 words**
- 最多 **10 illustrations**，即图和表总数原则上不超过 10
- 首选 Word，无强制 Word 模板；也接受 SAGE LaTeX 模板
- 摘要为非结构式摘要
- 至少 5 个关键词
- 参考文献采用 SAGE Vancouver 数字顺序制
- 单匿名审稿
- 文末必须包含 `Statements and Declarations`
- 可提交代码、补充实验和数据说明作为 supplementary material
- 标题应简洁、明确、可检索
- 投稿前核查 SAGE 最新 AI 使用披露政策

SAGE LaTeX/稿件指南入口：

- TIMC 官方投稿要求：https://journals.sagepub.com/author-instructions/tim
- SAGE 稿件准备指南：https://www.sagepub.com/journals/information-for-authors/preparing-your-manuscript
- **SAGE LaTeX 模板直接下载：https://uk.sagepub.com/sites/default/files/sage_latex_template_4.zip**
- SAGE 官方 Overleaf 模板列表（下载链接失效时使用）：https://www.overleaf.com/latex/templates/tagged/sage

### 1.4 模板下载与强制使用流程

Agent **不得先在空白 `.tex`、Markdown 或其他期刊模板中撰写论文**。在生成任何论文正文前，必须先完成以下步骤：

1. 访问 TIMC 官方投稿要求，确认该刊当前仍接受 LaTeX 文件；
2. 从上述直接下载链接下载最新版 SAGE LaTeX 模板；
3. 将模板压缩包和解压目录保存在项目中，不得只把模板当作网页参考；
4. 阅读模板中的示例 `.tex`、文档类 `.cls`、参考文献样式和说明文件；
5. 复制模板示例文件作为论文主文件，在该模板内部逐节撰写；
6. 首次写作前编译一次原始模板，确认模板可正常编译；
7. 每次形成完整初稿后重新编译，确保无缺失引用、未定义命令或图表溢出；
8. 不得自行重建一个“看起来像 SAGE”的模板，也不得删除模板要求的声明、文档类参数或投稿结构。

推荐执行命令：

```bash
mkdir -p paper/template
curl -L "https://uk.sagepub.com/sites/default/files/sage_latex_template_4.zip" \
  -o paper/template/sage_latex_template_4.zip
unzip -o paper/template/sage_latex_template_4.zip \
  -d paper/template/sage_latex
find paper/template/sage_latex -maxdepth 3 -type f | sort
```

若执行环境没有 `curl`，可使用：

```bash
wget -O paper/template/sage_latex_template_4.zip \
  "https://uk.sagepub.com/sites/default/files/sage_latex_template_4.zip"
```

模板安装后的最低目录要求：

```text
paper/
├── template/
│   ├── sage_latex_template_4.zip
│   └── sage_latex/                 # 官方压缩包解压结果
├── manuscript/
│   ├── main.tex                    # 由官方示例 tex 复制并修改
│   ├── references.bib
│   ├── figures/
│   └── tables/
└── template_check.md               # 记录下载日期、链接、模板文件和编译结果
```

`paper/template_check.md` 至少记录：

- 下载日期；
- 模板下载 URL；
- 模板压缩包文件名；
- 实际使用的主 `.tex` 示例文件；
- 使用的 `.cls` 和 `.bst` 文件；
- LaTeX 编译器及版本；
- 原始模板是否编译成功；
- Agent 是否基于该模板而非自行创建文档骨架。

若直接下载链接失效，Agent 必须：

1. 打开 SAGE 官方稿件准备指南；
2. 从页面中的 **Sage LaTeX Template** 链接获取当前模板，或从 SAGE 官方 Overleaf 模板列表复制项目；
3. 将实际下载链接和失效情况记录到 `paper/template_check.md`；
4. 完成模板下载和编译后才能继续撰写，不能跳过模板步骤。

SAGE 官方稿件准备页面明确要求，接受 LaTeX 的期刊应使用其 LaTeX Template；TIMC 官方投稿页也确认该刊接受 LaTeX，并指向 SAGE Journal Author Gateway 的模板。因此，Agent 必须把下载后的官方模板作为论文初稿的唯一排版起点。

### 1.5 篇幅规划

用户目标为约 16 页，但期刊的 6000 词限制优先于页数。Agent 应：

- 英文正文控制在 **5200–5800 words**；
- 参考文献、图注和声明另计时，也应尽量保持紧凑；
- 使用 SAGE 模板生成的工作稿预计约 13–16 页，实际页数由版式决定；
- 不得为了凑足 16 页而增加重复背景。

---

## 2. 论文题目与核心叙事

### 2.1 推荐题目

首选：

> **Delay-Aware Residual Offline Reinforcement Learning for Flow Tracking in Pipeline Systems**

备选：

> **DAR-TD3BC: Delay-Aware Residual Policy Learning for Offline Control of Delayed Pipeline Processes**

标题中避免直接写 “industrial deployment”“guaranteed stability” 或 “safe control”，因为当前实验仅使用公开仿真基准，且没有形式化稳定性证明。

### 2.2 研究问题

论文回答三个问题：

**RQ1.** 显式延迟表征是否优于将 25 步历史直接展平输入普通 TD3+BC？

**RQ2.** 行为先验残差策略与不确定性门控是否能在不产生明显分布外动作的情况下改善跟踪性能？

**RQ3.** 方法是否能同时提升标准化回报和控制领域指标，并在观测噪声、数据减少与延迟变化下保持更好的鲁棒性？

### 2.3 核心论证链

全文必须围绕以下单一主线展开：

1. 离线 RL 适合不能任意在线探索的控制系统；
2. Pipeline 同时具有离线分布偏移和长动作响应延迟；
3. 原始基准把历史直接拼接，通用离线 RL 没有显式学习时延因果关系；
4. 单纯换成 GRU/Transformer 不足以形成有说服力的创新；
5. 因此设计“预测辅助的延迟表征 + 行为先验残差策略 + 不确定性自适应门控”；
6. 用官方回报、控制指标与鲁棒性实验验证完整方法的整体效果；
7. 结论限定为“在 NeoRL-2 Pipeline 仿真基准上有效”，不外推为真实工厂验证。

---

## 3. 方法定义

### 3.1 状态与动作

NeoRL-2 Pipeline 的观测维数为 52，动作维数为 1。按官方环境源码解析为：

\[
o_t=
[y_t,\ r_t,\ y_{t-1:t-25},\ a_{t-1:t-25}],
\]

其中：

- \(y_t\)：当前下游流量；
- \(r_t\)：目标流量；
- \(a_t\in[-1,1]\)：入口水门增量动作；
- 历史序列用于描述延迟传播。

Agent 必须核对官方源码中的历史顺序。当前源码通过头部插入保存历史，因此数组通常为“最近到最远”；送入时序编码器前应转换为时间正序。

还必须区分：

- 论文/数据卡描述的 25 步历史和延迟设置；
- 当前仿真器默认 `channel_length=100`、`pipe_velocity=5`，物理运输队列长度为 `ceil(100/5)=20` 步。

不得把“历史窗口 25”和“源码默认运输队列 20”混为一谈。正文应在实验设置中准确说明版本和源码行为。

### 3.2 延迟编码器

将历史转换为：

\[
X_t =
[(y_{t-25},a_{t-25}),\ldots,(y_{t-1},a_{t-1})].
\]

采用轻量级 causal TCN 或 GRU：

\[
z_t=f_\phi(X_t,y_t,r_t).
\]

首选 TCN，因为：

- 并行训练效率较高；
- 感受野容易覆盖固定历史窗口；
- 因果卷积符合控制时序；
- 参数量较小，便于做实时推理分析。

若代码实现优先使用 GRU 或 TCN，必须按实际实现如实描述；本课程论文不要求额外比较 MLP、GRU、TCN 编码器变体。

### 3.3 多步预测辅助任务

利用离线轨迹构造多步流量目标：

\[
\hat y_{t+h}=g_{\psi,h}(z_t),\quad
h\in\{1,5,10,20\}.
\]

辅助损失：

\[
\mathcal L_{\mathrm{pred}}
=
\sum_{h\in\mathcal H}
w_h\left\|\hat y_{t+h}-y_{t+h}\right\|_2^2.
\]

作用：

- 迫使表示编码传播延迟；
- 减少仅靠回报学习导致的弱时序监督；
- 为“延迟感知”提供可直接验证的辅助指标。

未来目标不得跨越 episode 边界。

### 3.4 行为先验与残差策略

先训练行为策略：

\[
a_t^{\mathrm{bc}}=\beta_\omega(o_t).
\]

残差 actor 输出：

\[
\delta a_t=\pi_\theta(z_t).
\]

最终动作：

\[
a_t^\pi=
\operatorname{clip}
\left(
a_t^{\mathrm{bc}}+
g_t\delta a_t,\,-1,\,1
\right).
\]

该结构把策略改进限制在数据支持区域附近，比直接输出完整动作更适合保守的离线控制数据。

### 3.5 不确定性门控

使用双 critic 的分歧估计局部不确定性：

\[
u_t=
|Q_{\varphi_1}(z_t,a_t)-Q_{\varphi_2}(z_t,a_t)|.
\]

定义门控：

\[
g_t=\exp(-\kappa\,\operatorname{stopgrad}(u_t)),
\]

或使用经过归一化和截断的等价形式。其作用是：

- critic 一致时允许较大的残差修正；
- critic 分歧大时自动回退到行为先验。

未进行消融时，Agent 只能将门控作为完整方法的一部分解释其设计动机，不得声称其单独有效或“保证安全”。

### 3.6 损失函数

TD3 critic：

\[
y_t^{Q}
=
r_t+
\gamma(1-d_t)
\min_{i=1,2}
Q_{\bar\varphi_i}
(z_{t+1},\tilde a_{t+1}),
\]

其中目标动作使用 clipped target noise。

Actor 损失：

\[
\mathcal L_{\mathrm{actor}}
=
-\lambda_Q
\mathbb E[
\min(Q_{\varphi_1},Q_{\varphi_2})(z_t,a_t^\pi)
]
+
\alpha_{\mathrm{bc}}
\mathbb E[\|a_t^\pi-a_t^{D}\|_2^2]
+
\alpha_{\mathrm{res}}
\mathbb E[\|\delta a_t\|_2^2]
+
\alpha_{\mathrm{tv}}
\mathbb E[\|a_t^\pi-a_{t-1}^{D}\|_2^2].
\]

总训练目标可写为：

\[
\mathcal L
=
\mathcal L_{\mathrm{critic}}
+
\mathcal L_{\mathrm{actor}}
+
\alpha_{\mathrm{pred}}\mathcal L_{\mathrm{pred}}.
\]

主实验保持官方环境奖励不变。动作平滑通过 actor 正则项实现，避免因改奖励而破坏官方回报的可比性。

---

## 4. 实验设计

### 4.1 数据和环境

数据与环境必须来自官方来源：

- NeoRL-2 GitHub: https://github.com/polixir/NeoRL2
- Hugging Face: https://huggingface.co/datasets/polixirai/NeoRL2
- 论文: https://arxiv.org/abs/2503.19267

官方 Python 接口：

```python
import neorl2
import gymnasium as gym

env = gym.make("Pipeline")
train_data, val_data = env.get_dataset()
```

必须记录：

- NeoRL-2 commit SHA；
- Python、PyTorch、Gymnasium 版本；
- 数据集哈希；
- 训练集和验证集实际样本数；
- 观测、动作、奖励和终止标记的实际形状；
- 归一化统计量只由训练集计算。

### 4.2 基线

分为两组。

#### A. 官方已报告基线

从 NeoRL-2 原论文表格中录入：

- Data policy
- BC
- CQL
- EDAC
- MCQ
- TD3+BC
- MOPO
- COMBO
- RAMBO
- MOBILE

这些数据可以不复现，但必须：

- 标注 `Reported by Gao et al. (2025)`；
- 精确注明表号、指标、均值、误差类型和种子数；
- 不与本地结果混写成同一实验；
- 不复制论文文字，只录入合法引用的数值；
- 在表下注明硬件、代码版本和随机种子可能不同，因此只用于背景比较。

根据当前论文表格，应核验并录入以下 Pipeline 结果：

| Method | Normalized return |
|---|---:|
| Data policy | 69.25 |
| BC | 68.61 ± 13.42 |
| CQL | 81.08 ± 8.25 |
| EDAC | 72.88 ± 4.64 |
| MCQ | 49.70 ± 7.39 |
| TD3+BC | 81.95 ± 7.46 |
| MOPO | -26.33 ± 92.65 |
| COMBO | 55.50 ± 4.28 |
| RAMBO | 24.06 ± 74.39 |
| MOBILE | 65.51 ± 4.05 |

Agent 必须回到原文核实误差是标准差还是标准误，不得仅依据本文盲目引用。

#### B. 本项目统一实现的内部主比较

至少运行：

1. `TD3BC-MLP`：同代码框架下的普通 TD3+BC；
2. `DAR-TD3BC`：完整方法。

内部主比较用于给出同代码框架下的公平基线，官方报告值用于外部参照。不能只和文献数值比较而没有任何同代码基线。由于时间成本限制，本课程论文不执行消融实验，也不在正文中报告或声称单模块消融结论。

### 4.3 主评价指标

官方指标：

- NeoRL-2 normalized return；
- raw episodic return。

控制指标：

\[
\mathrm{RMSE}
=
\sqrt{\frac1T\sum_t(y_t-r_t)^2},
\]

\[
\mathrm{MAE}
=
\frac1T\sum_t|y_t-r_t|,
\]

\[
\mathrm{IAE}
=
\sum_t |y_t-r_t|\Delta t.
\]

另报告：

- target-switch settling time；
- maximum overshoot；
- steady-state error；
- action total variation \(\sum_t|a_t-a_{t-1}|\)；
- action energy \(\sum_t a_t^2\)；
- 单步推理延迟和参数量。

调节时间预先定义为：目标变化后，误差进入并连续 10 步保持在  
\(\max(0.02|r_t|,2.0)\) 范围内的首个时刻。定义一旦确定，不得根据结果更换。

### 4.4 统计设置

最低要求：

- 5 个独立训练种子；
- 每个 checkpoint 至少评估 20 个 episode；
- 报告 mean ± standard deviation 和 95% bootstrap confidence interval；
- 同一组环境种子用于方法间配对比较；
- 主要方法与 TD3BC-MLP 使用配对 Wilcoxon 检验或配对 bootstrap；
- 同时报告效应量；
- 不以单一最好 seed 作为最终结果。

### 4.5 鲁棒性实验

至少选择三个维度：

1. 数据比例：25%、50%、75%、100%；
2. 归一化观测噪声标准差：0、0.01、0.03、0.05；
3. 物理运输延迟：通过参数化 `channel_length/pipe_velocity` 构造 10、15、20、25、30 步；
4. 目标切换概率：0.001、0.003、0.005、0.01。

修改环境时必须创建独立 wrapper 或 fork 文件，不得直接无记录修改第三方源码。默认环境结果和改动环境结果必须分开报告。

### 4.6 图表额度

总图表不超过 10，建议使用：

1. Figure 1：Pipeline 控制任务和延迟作用示意；
2. Figure 2：DAR-TD3BC 网络结构与训练流程；
3. Figure 3：典型目标切换下的流量跟踪和动作曲线，多子图合并为一个图；
4. Figure 4：主结果与控制指标，多面板；
5. Figure 5：延迟、噪声和数据比例鲁棒性热图；
6. Table 1：任务、数据和训练设置；
7. Table 2：官方报告基线，带 provenance；
8. Table 3：本项目统一实现主结果；
9. Table 4：统计检验、参数量与推理延迟。

补充材料放置完整学习曲线、全部 seed、更多轨迹和超参数。

---

## 5. 论文各部分写作要求

## Title page

包含：

- 论文题目；
- 作者与单位；
- 通讯作者；
- ORCID；
- word count；
- figure/table count；
- code and data availability statement。

初稿阶段不得编造作者顺序和基金号，使用占位符 `[AUTHOR]`、`[AFFILIATION]`、`[FUNDING]`。

## Abstract：180–220 words

按一段写完，必须包含：

1. 背景：离线控制避免昂贵或危险的在线探索；
2. 问题：动作响应延迟与离线分布偏移同时存在；
3. 方法：预测辅助时序编码、行为先验残差策略、不确定性门控；
4. 数据：NeoRL-2 Pipeline；
5. 结果：只能从 `results/final_summary.csv` 自动填入真实数值；
6. 结论：说明方法在公开延迟过程控制基准上的效果；
7. 不得写“guarantees stability/safety”；
8. 不得在实验尚未完成时填入假结果。

结果未完成时使用：

`[RESULT_RMSE]`、`[RESULT_RETURN]`、`[RESULT_SIGNIFICANCE]`

并在文档顶部显示 `DRAFT WITH UNRESOLVED PLACEHOLDERS`。

## Keywords：5–7 个

建议：

- offline reinforcement learning
- delayed control systems
- data-driven control
- residual policy learning
- TD3+BC
- pipeline flow control
- uncertainty-aware control

## 1. Introduction：850–950 words

写作顺序：

1. 工业过程控制中时延、在线试错成本和历史日志利用的现实矛盾；
2. 离线 RL 的优势及其分布外动作问题；
3. 延迟导致观测不再充分满足马尔可夫性和信用分配困难；
4. NeoRL-2 Pipeline 的代表性和现有通用基线不足；
5. 现有简单做法：历史展平、RNN 表征、保守离线 RL；
6. 明确研究缺口：缺少同时结合延迟表征、受限策略改进和控制平滑性的完整方法；
7. 给出 3 条贡献，不能超过 4 条；
8. 简述全文结构。

推荐贡献表述框架：

- 提出一种由多步预测监督的延迟感知表示，以学习历史动作与未来流量之间的传播关系；
- 提出行为先验残差 actor 与 critic-disagreement gate，在离线数据支持范围内进行自适应策略改进；
- 在 NeoRL-2 Pipeline 上按标准回报和控制指标进行统一评估，并通过扰动实验分析完整方法的鲁棒性。

不要把“首次”写入贡献，除非完成系统文献检索并有充分证据。

## 2. Related Work：750–850 words

分成三小节：

### 2.1 Offline reinforcement learning for continuous control

覆盖：

- BC；
- TD3+BC；
- CQL；
- IQL；
- ensemble/conservative methods；
- model-based offline RL。

重点不是列算法，而是比较“如何限制 OOD 动作”。

### 2.2 Reinforcement learning for delayed systems

覆盖：

- augmented-state MDP；
- recurrent state estimation；
- delay-aware credit assignment；
- belief-state/Transformer methods；
- delayed offline RL。

必须检索并讨论：

- Random Delays in Reinforcement Learning；
- Delay-Correcting Actor-Critic；
- DEER；
- DT-CORL；
- 2025–2026 年 delayed offline RL 最新工作。

### 2.3 Residual and learning-based process control

覆盖：

- residual RL；
- policy correction around PID/MPC/behavior policies；
- data-driven process control；
- Smith predictor/MPC 处理时延的基本思想。

结尾用表格或一段明确区分：

| Method family | Delay representation | Offline support constraint | Residual prior | Control-oriented evaluation |
|---|---|---|---|---|

不得只说前人“没有考虑”，必须给出可核查差异。

## 3. Problem Formulation：500–600 words

包含：

- delayed MDP / partially observable formulation；
- 离线数据集定义；
- Pipeline 状态、动作、转移、奖励；
- 官方奖励函数；
- 训练时不能在线探索，评估时只允许 policy rollout；
- 优化目标；
- 控制评价指标；
- 历史窗口 25 与源码运输队列 20 的版本说明。

官方奖励必须从当前 commit 源码重新核验。当前已知形式近似为：

\[
r_t =
\left(0.01(200-|y_t-r_t|)\right)^2-3.
\]

不得在主实验中自行替换。

## 4. Proposed Method：1200–1350 words

建议结构：

### 4.1 Overview
结合 Figure 2 解释训练和推理数据流。

### 4.2 Delay-aware representation
给出时序输入、TCN/GRU 结构和多步预测损失。

### 4.3 Behavior-prior residual actor
给出 BC 预训练和残差动作。

### 4.4 Uncertainty-adaptive gate
给出 critic 分歧、门控定义和直观解释。

### 4.5 Offline actor–critic optimization
给出 critic target、actor loss、target smoothing 和 delayed policy update。

### 4.6 Training procedure
给出 Algorithm 1 伪代码。伪代码应控制在一个栏宽或作为补充材料，避免占用过多图表额度。

只解释方法所需内容，不写教材式 TD3 背景。

## 5. Experimental Setup：650–750 words

必须写明：

- 数据版本和下载方式；
- 数据划分；
- 网络规模；
- 归一化；
- optimizer；
- batch size；
- discount；
- target update；
- policy delay；
- target noise；
- 训练步数；
- seed；
- checkpoint selection；
- 硬件；
- 评估 episode 数；
- 基线来源和 provenance；
- 控制指标定义；
- 统计检验；
- 鲁棒性环境改动。

超参数表放在 Table 1 或补充材料。

## 6. Results：1050–1200 words

按问题而不是按图编号组织：

### 6.1 Overall performance
先报告内部公平基线，再引用官方报告基线。说明均值、方差、置信区间和实际改进量。

### 6.2 Tracking dynamics and control effort
分析典型目标切换轨迹、超调、调节时间、动作抖动。

### 6.3 Robustness
报告噪声、数据比例和延迟变化。

### 6.4 Computational cost
报告参数量、训练时间、单步推理时间。

结果段不得只说 “significantly better”；必须给出数字和统计证据。任何数字都必须追溯到 CSV/JSON。

## 7. Discussion and Limitations：350–450 words

必须包含：

- 方法为什么对延迟任务有效；
- 与 Smith predictor、MPC 的关系；
- 离线行为先验的优势和局限；
- 仿真基准与真实管道之间的差距；
- 历史窗口和运输模型较简化；
- critic disagreement 只是经验不确定性代理；
- 没有闭环稳定性严格证明；
- 后续需要真实过程数据、硬件在环或形式化约束。

## 8. Conclusion：180–220 words

只写：

- 问题；
- 方法；
- 最关键真实结果；
- 适用范围；
- 一项明确未来工作。

不得重复摘要或列出所有实验。

## Statements and Declarations

按 SAGE 最新要求保留全部小标题。至少包括：

- Ethical considerations
- Consent to participate
- Consent for publication
- Declaration of conflicting interests
- Funding
- Author contributions
- Data availability
- Code availability
- Acknowledgements
- AI assistance disclosure（按投稿时政策）

对于纯仿真研究，不适用项写 `Not applicable`，不能直接删除标题。

---

## 6. 文献检索任务

Agent 必须先生成 `paper/evidence/source_ledger.csv`，字段：

```text
citation_key,title,authors,year,venue,doi,url,source_type,claim_supported,checked_date,notes
```

优先级：

1. 原始论文；
2. 官方代码与数据卡；
3. 官方期刊指南；
4. 高质量综述；
5. 不使用博客作为技术结论依据。

### 6.1 必查资料

任务与数据：

- NeoRL-2 paper: https://arxiv.org/abs/2503.19267
- NeoRL-2 code: https://github.com/polixir/NeoRL2
- NeoRL-2 data: https://huggingface.co/datasets/polixirai/NeoRL2

算法：

- TD3+BC paper: https://arxiv.org/abs/2106.06860
- TD3+BC code: https://github.com/sfujim/TD3_BC
- CQL: https://arxiv.org/abs/2006.04779
- IQL: https://arxiv.org/abs/2110.06169
- OfflineRL library: https://github.com/polixir/OfflineRL
- CORL library: https://github.com/tinkoff-ai/CORL

时延强化学习：

- https://arxiv.org/abs/2108.07555
- https://arxiv.org/abs/2010.02966
- https://arxiv.org/abs/2406.03102
- https://arxiv.org/abs/2506.00131

检索式：

```text
"offline reinforcement learning" delayed control system
"offline RL" action delay process control
"delay-aware" offline reinforcement learning
"residual reinforcement learning" process control
"behavior prior" offline RL continuous control
"uncertainty gating" offline reinforcement learning
"pipeline flow control" reinforcement learning
"Smith predictor" reinforcement learning time delay
site:journals.sagepub.com/doi "offline reinforcement learning" control
site:journals.sagepub.com/doi "reinforcement learning" "Transactions of the Institute of Measurement and Control"
```

检索时间范围以 2020–投稿当年为主，同时保留必要的经典控制和 RL 原始文献。

### 6.2 新颖性检查

在写引言和贡献前，Agent 必须回答：

- 是否已有“GRU/Transformer + TD3+BC”用于 delayed offline control？
- 是否已有 behavior-prior residual TD3+BC？
- 是否已有 critic disagreement gate 控制残差幅度？
- 是否已有同方法在 NeoRL-2 Pipeline 上发表？

将结果写入 `paper/novelty_audit.md`。发现高度相似方法时，必须调整贡献，不得隐瞒。

---

## 7. Agent 自动生成初稿的执行顺序

1. 读取本文件和代码实施文件；
2. 核验期刊当前范围、字数、图表、参考文献和声明要求；
3. 核验 JCR/SCIE 状态；
4. 建立 source ledger；
5. 读取所有实验配置、日志、CSV、图和统计结果；
6. 检查每个结果是否有 provenance；
7. 先生成详细英文提纲；
8. 再生成各节英文初稿；
9. 使用真实结果替换占位符；
10. 自动检查总词数、图表数和引用顺序；
11. 生成 Word 优先版本，同时保留 LaTeX 工作版本；
12. 生成 cover letter 和 supplementary material；
13. 输出未解决问题清单，不得自行编造。

### 7.1 需要生成的文件

```text
paper/
├── manuscript.docx
├── main.tex
├── references.bib
├── cover_letter.md
├── highlights.md
├── novelty_audit.md
├── journal_check.md
├── unresolved_items.md
├── evidence/
│   └── source_ledger.csv
└── supplementary/
    ├── supplementary.tex
    ├── hyperparameters.md
    └── complete_results.csv
```

### 7.2 写作约束

- 正文使用学术英语；
- 逻辑顺序为“问题—缺口—方法—证据—边界”；
- 不使用夸张词 `groundbreaking`、`revolutionary`；
- 不写无法证实的 `first`、`guarantee`、`real-world deployment`；
- 不从参考论文复制成段文字；
- 每个具体事实、数据和对比都必须有引用或项目结果文件；
- 报告负结果和失败设置；
- 结果完成前保留显眼占位符，不得生成看似真实的数字；
- 对官方报告结果和本地复现实验使用不同表格列和脚注；
- 论文中所有代码链接在匿名审稿阶段按期刊要求处理。

### 7.3 初稿验收标准

只有同时满足以下条件，初稿才算完成：

- 5200–5800 英文词；
- 图表总数不超过 10；
- 至少 5 个关键词；
- 所有 section 与期刊要求一致；
- 所有结果均可追溯到 `results/`；
- 至少 5 个训练种子的统计；
- 包含公平内部 TD3+BC 基线；
- 包含鲁棒性和计算成本；
- 包含完整限制；
- 无虚构引用；
- 无未标记的占位符；
- `Statements and Declarations` 完整；
- 通过拼写、引用、图号、表号和公式编号检查。
