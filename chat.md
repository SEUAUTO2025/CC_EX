# 开展论文与代码实现

> 先大致浏览01_论文写作方案_TIMC_DAR_TD3BC.md，了解我们的目标：形成16页左右的期刊论文以及其大致思路，然后仔细研读02_代码实施方案_DAR_TD3BC.md理清具体代码实现逻辑。之后开始进行代码撰写（建一个code文件夹，所有代码都放到那里面）以及论文写作（使用academic-research-suite进行研究工作的全套流程，这个skill是本次流程中的重要参考，要认真遵守（找不到的话先停止工作向我汇报））。过程中有需要我补充的文件或是任何疑问，使用grill-with-docs向我提问

<details><summary>5 previous messages</summary>

> 我会先确认必须使用的 skill 是否可用。你指定找不到 `academic-research-suite` 就暂停，所以在确认之前我不会改代码或论文文件。
>
> 我已加载 `superpowers:using-superpowers` 的流程要求。现在只做一件事：在本机 skill 目录里查找 `academic-research-suite`，确认它是否确实可用。
>
> 找到了本地目录 `C:\Users\SEUAUTO_Zhang\.codex\skills\academic-research-suite`。我先完整读取它的 `SKILL.md`，再按它的流程推进。
>
> 目录存在，但标准入口 `SKILL.md` 不存在。我再看一下这个目录结构；如果没有可用入口文件，就按你的要求暂停汇报。
>
> 这里已经形成一个阻塞点：`academic-research-suite` 有目录但不是可加载的标准 skill。你要求有疑问用 `grill-with-docs`，我先加载该 skill 后再把问题整理给你。

</details>

我已按你的要求暂停。

`academic-research-suite` 没有作为可用 skill 出现在当前 skill 列表里。进一步检查后，发现本地确实有目录：

`C:\Users\SEUAUTO_Zhang\.codex\skills\academic-research-suite`

但其中没有标准入口文件 `SKILL.md`，只有 `README.md`、`agents/`、`hooks/`、`scripts/` 等内容。因此我现在不能把它作为 Codex skill 正常加载和遵守。尚未开始浏览 `01_论文写作方案_TIMC_DAR_TD3BC.md`、`02_代码实施方案_DAR_TD3BC.md`，也没有创建或修改任何代码/论文文件。

按 `grill-with-docs` 的方式先问一个阻塞问题：

你希望我如何处理这个非标准的 `academic-research-suite`？

我的推荐答案：请你补充或恢复该目录下的 `SKILL.md`，让我能按标准 skill 入口执行；如果你确认允许，我也可以把该目录的 `README.md` 当作流程参考继续，但这不等价于“使用该 skill”。

> 已经补充

<details><summary>63 previous messages</summary>

> 我会重新读取 `academic-research-suite` 的入口文件，并同时加载本次实现必须用到的流程类 skill：`writing-plans` 和 `test-driven-development`。之后再开始读你的两份方案文档。
>
> `academic-research-suite` 现在可以加载，入口正常。当前请求会走 ARS 的 `academic-pipeline` 工作流；代码实现部分我会按 `writing-plans` 和 TDD 执行，论文部分再按 ARS 的论文/实验阶段约束推进。
>
> `academic-pipeline` 路径在当前安装里和入口文件描述不一致，我正在定位实际 vendored ARS 目录。已经确认根目录只有两份方案文档和一份研究报告，接下来会先把路径问题理清。
>
> ARS 根入口指向的 `ars/*/WORKFLOW.md` 在这份安装里缺失，但它提供了 Codex adapter 的 agent/manifest 文件。我会按入口文件的 fallback 规则报告“降级为 inline role-prompt execution”，读取可用的 orchestrator/academic-paper agent 后继续，不擅自假装完整 ARS upstream 已加载。
>
> 论文方案已经读到关键结构，但之前的终端编码把中文显示成乱码。我正在用 UTF-8 重新读取两份方案，这样后续代码和论文不会建立在误读之上。
>
> 我已经理清两份方案的主线：先搭一个可测试的研究代码库，至少跑通数据接口/观测解析/模型与更新逻辑的 smoke path；论文侧必须先完成官方模板和证据台账，未出真实实验结果前只能放显式占位符。接下来我会先做实现计划和项目骨架，然后按 TDD 写核心代码。
>
> 在写文件前我会先看运行环境、Git 状态和可用工具，避免把项目结构建在不存在的依赖上。之后会把计划保存到 `docs/superpowers/plans/`，再直接进入 inline 执行。
>
> 当前目录不是 Git 仓库，Python 是 3.12.7，不是方案推荐的 3.10。我会先让代码支持可导入和单元测试；涉及 NeoRL-2 的真实训练会在文档和脚本中记录 Python 3.10 约束，避免把 3.12 当成已验证实验环境。
>
> `gymnasium` 当前环境未安装，Torch/Pandas 导入比默认超时慢，我会用更长超时确认。核心单测会尽量不依赖 NeoRL-2 在线环境；训练脚本会在依赖缺失时给出明确错误和数据请求路径。
>
> 接下来会写第一批文件：一个执行计划、Python 项目元数据、测试文件和只抛 `NotImplementedError` 的最小接口桩。这样第一次 pytest 会证明测试确实在约束未实现行为。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./docs/superpowers/plans/2026-06-15-dar-td3bc-research-pipeline.md` (+149 -0)
>
> </details>
>
> 收到。环境部分我只写成说明/记录文本，不把当前电脑当成实验环境来跑真实实验；现在优先写 `code/` 里的代码，并用本地单元测试验证核心逻辑。
>
> 我正在使用 `executing-plans` 执行刚才的计划。这个目录不是 Git 仓库，所以不会创建 worktree 或分支；我会直接在当前工作区建 `code/`，不碰你的两份方案文档。
>
> 我现在开始第一轮 TDD：先放测试和接口桩，不写实现。随后跑一次 pytest，确认失败原因是“功能未实现”，再补实现。
>
> <details><summary>Created 13 files</summary>
>
> - Wrote `./code/pyproject.toml` (+22 -0)
> - Wrote `./code/src/dar_td3bc/__init__.py` (+5 -0)
> - Wrote `./code/src/dar_td3bc/data/__init__.py` (+1 -0)
> - Wrote `./code/src/dar_td3bc/data/pipeline_obs.py` (+22 -0)
> - Wrote `./code/src/dar_td3bc/data/trajectory_index.py` (+30 -0)
> - Wrote `./code/src/dar_td3bc/evaluation/__init__.py` (+1 -0)
> - Wrote `./code/src/dar_td3bc/evaluation/control_metrics.py` (+45 -0)
> - Wrote `./code/src/dar_td3bc/models/__init__.py` (+1 -0)
> - Wrote `./code/src/dar_td3bc/models/temporal_encoder.py` (+35 -0)
> - Wrote `./code/tests/test_control_metrics.py` (+45 -0)
> - Wrote `./code/tests/test_obs_parser.py` (+33 -0)
> - Wrote `./code/tests/test_temporal_encoder.py` (+24 -0)
> - Wrote `./code/tests/test_trajectory_index.py` (+36 -0)
>
> </details>
>
> 接口桩和第一批测试已经放进 `code/`。我现在跑一次测试，确认它们按预期失败，然后再补实现。
>
> RED 阶段通过：8 个测试都失败在 `NotImplementedError` 或形状校验未实现上，说明测试确实覆盖了待写行为。现在补观测解析、轨迹边界、控制指标和 TCN 编码器。
>
> <details><summary>Edited 4 files</summary>
>
> - Wrote `./code/src/dar_td3bc/data/pipeline_obs.py` (+22 -1)
> - Wrote `./code/src/dar_td3bc/data/trajectory_index.py` (+58 -2)
> - Wrote `./code/src/dar_td3bc/evaluation/control_metrics.py` (+83 -6)
> - Wrote `./code/src/dar_td3bc/models/temporal_encoder.py` (+53 -3)
>
> </details>
>
> 核心解析、边界 mask、控制指标和编码器已经实现。现在跑对应测试，若有测试本身对边界定义不严谨，我会先修正测试的期望再继续。
>
> 一个测试期望写错了：第二条轨迹是索引 5 到 9，`horizon=5` 会访问索引 10，已经越界，不应有效。我会把这个测试改成期望 mask 为 0，保持“不跨轨迹也不越界”的定义。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/tests/test_trajectory_index.py` (+1 -1)
>
> </details>
>
> 测试期望已修正。现在重新跑第一批核心测试，确认 GREEN。
>
> 第一批 8 个核心测试已经通过。下一步继续 TDD：先写策略、critic、预测头和 gate 的测试与接口桩，再跑 RED。
>
> <details><summary>Created 3 files</summary>
>
> - Wrote `./code/src/dar_td3bc/models/critic.py` (+20 -0)
> - Wrote `./code/src/dar_td3bc/models/policies.py` (+31 -0)
> - Wrote `./code/tests/test_models.py` (+81 -0)
>
> </details>
>
> 策略和 critic 测试已写好，接口仍是桩。现在跑这组测试，确认它们先失败。
>
> RED 确认完成，6 个模型测试都失败在未实现接口上。现在写最小 PyTorch 实现。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./code/src/dar_td3bc/models/critic.py` (+32 -2)
> - Wrote `./code/src/dar_td3bc/models/policies.py` (+39 -4)
>
> </details>
>
> 模型实现已补齐。现在跑全部代码测试，确认核心代码没有破坏已通过的部分。
>
> 核心模型单测 14 个全部通过。为了让后续训练脚本有可复用基础，我继续补训练所需的纯函数层：归一化、masked prediction loss、TD3+BC actor loss、Polyak target update 和随机种子工具。
>
> <details><summary>Created 8 files</summary>
>
> - Wrote `./code/src/dar_td3bc/algorithms/__init__.py` (+1 -0)
> - Wrote `./code/src/dar_td3bc/algorithms/td3bc.py` (+41 -0)
> - Wrote `./code/src/dar_td3bc/data/normalization.py` (+25 -0)
> - Wrote `./code/src/dar_td3bc/utils/__init__.py` (+1 -0)
> - Wrote `./code/src/dar_td3bc/utils/seed.py` (+5 -0)
> - Wrote `./code/tests/test_losses.py` (+61 -0)
> - Wrote `./code/tests/test_normalization.py` (+31 -0)
> - Wrote `./code/tests/test_reproducibility.py` (+16 -0)
>
> </details>
>
> 新一批训练基础测试和接口桩已添加。现在先跑这些测试，确认 RED。
>
> RED 确认完成，6 个训练基础测试都失败在未实现接口上。现在补最小实现。
>
> <details><summary>Edited 3 files</summary>
>
> - Wrote `./code/src/dar_td3bc/algorithms/td3bc.py` (+30 -3)
> - Wrote `./code/src/dar_td3bc/data/normalization.py` (+11 -4)
> - Wrote `./code/src/dar_td3bc/utils/seed.py` (+12 -1)
>
> </details>
>
> 归一化、loss、soft update 和 seed 工具已实现。现在跑完整测试集。
>
> 完整测试现在 20 个通过。接着补代码侧的配置与 provenance 工具：官方 reported baseline 只作为“未复跑、待原文核验”的 CSV 种子数据；实验环境按你的要求写成 txt，不要求在当前电脑真实跑训练。
>
> <details><summary>Created 5 files</summary>
>
> - Wrote `./code/configs/model/dar_td3bc.yaml` (+37 -0)
> - Wrote `./code/scripts/import_reported_baselines.py` (+22 -0)
> - Wrote `./code/scripts/write_environment_txt.py` (+22 -0)
> - Wrote `./code/src/dar_td3bc/utils/provenance.py` (+126 -0)
> - Wrote `./code/tests/test_provenance.py` (+36 -0)
>
> </details>
>
> 我刚才把 provenance 测试和实现放在同一个补丁里了，这不符合本轮 TDD 纪律。这个代码是我刚写的，我会先把实现退回接口桩，跑 RED，再重新补实现。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/src/dar_td3bc/utils/provenance.py` (+4 -65)
>
> </details>
>
> provenance 实现已退回桩。现在跑对应测试，确认它们先失败。
>
> RED 已确认，3 个 provenance 测试都失败在未实现接口。现在重新补实现。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/src/dar_td3bc/utils/provenance.py` (+65 -4)
>
> </details>
>
> provenance 实现已补回。现在跑完整测试集。
>
> 完整测试现在 23 个通过。接下来生成代码侧的环境 txt 和 reported baseline CSV，确保这些脚本路径可用；然后我再补 README 和下一步训练入口说明。
>
> 脚本直接运行时找不到 `src` 包，这不是算法问题，是脚本入口缺少本地 `src` 路径。按调试流程我先确认根因，再改两个脚本的入口路径处理。
>
> 根因已确认：`pytest` 通过 `pyproject.toml` 把 `src` 加进路径，但直接执行 `code/scripts/*.py` 时，Python 只看到 `code/scripts`，看不到 `code/src`。我会先加一个子进程测试复现这个入口问题，再修脚本。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./code/tests/test_scripts.py` (+42 -0)
>
> </details>
>
> 子进程脚本测试已添加。现在跑它，确认失败与手动复现一致。
>
> 入口测试复现了相同的 `ModuleNotFoundError`。现在只修脚本启动路径，不改业务逻辑。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./code/scripts/import_reported_baselines.py` (+6 -0)
> - Wrote `./code/scripts/write_environment_txt.py` (+6 -0)
>
> </details>
>
> 脚本路径修复已完成。现在重新跑脚本测试和完整测试。
>
> 脚本入口问题已修复，完整测试 25 个通过。现在生成 `EXPERIMENT_ENVIRONMENT.txt` 和 reported baseline CSV。
>
> 我会再补一个数据获取入口：当前机器没有 `gymnasium/neorl2`，所以脚本应明确退出并打印需要用户提供的数据字段，而不是含糊报错。先写测试和桩。
>
> <details><summary>Created 3 files</summary>
>
> - Wrote `./code/scripts/acquire_data.py` (+24 -0)
> - Wrote `./code/src/dar_td3bc/data/acquisition.py` (+5 -0)
> - Wrote `./code/tests/test_acquire_data_script.py` (+44 -0)
>
> </details>
>
> 数据获取入口测试与桩已添加。现在跑它，确认 RED。
>
> RED 确认完成。现在实现清晰的数据缺失请求和自动获取尝试；当前环境缺少依赖时会返回退出码 2。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./code/scripts/acquire_data.py` (+18 -2)
> - Wrote `./code/src/dar_td3bc/data/acquisition.py` (+56 -1)
>
> </details>
>
> 数据获取脚本已实现。现在跑该测试和完整测试。
>
> 数据获取入口已通过测试，完整测试 27 个通过。现在补一份 `code/README.md`，把当前代码能做什么、如何测试、哪些是真实验前置项写清楚。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./code/README.md` (+70 -0)
>
> </details>
>
> README 已补上。现在做一次结构检查和最终测试，确认 `code/` 内产物完整。
>
> 测试生成了 `__pycache__` 缓存目录，我会清理这些生成物，保留源码、测试、配置和结果 CSV。
>
> 缓存已清理。为了最终验证不再留下缓存，我会用 `PYTHONDONTWRITEBYTECODE=1` 再跑一次测试。
>
> 代码侧第一版 scaffold 已完成并验证，且没有留下 `__pycache__`。我会更新任务状态，然后给你简短说明当前完成范围和下一步。

</details>

先按“代码优先”完成了第一版 `code/` 工程骨架，没有继续写论文正文。

已完成的主要内容：

- 核心代码在 [code/src/dar_td3bc](./code/src/dar_td3bc)
- 测试在 [code/tests](./code/tests)
- 配置在 [code/configs/model/dar_td3bc.yaml](./code/configs/model/dar_td3bc.yaml)
- 使用说明在 [code/README.md](./code/README.md)
- 实验环境说明 txt 在 [code/EXPERIMENT_ENVIRONMENT.txt](./code/EXPERIMENT_ENVIRONMENT.txt)

已实现：

- Pipeline 52 维观测解析与历史顺序翻转
- episode-safe 多步预测 target/mask
- RMSE/MAE/IAE、动作 TV/energy、settling time 指标
- TCN delay encoder、behavior policy、residual actor、twin critic、prediction head、critic-disagreement gate
- TD3+BC 相关 helper：soft update、masked prediction loss、actor loss components
- 数据获取脚本：缺 NeoRL-2/Gymnasium 时会明确列出所需数据字段
- reported baseline CSV 生成脚本，全部标记为 `reported_not_rerun` 和 `unverified_error_bar`

验证结果：

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q
27 passed
```

注意：这还是“可测试代码 scaffold”，不是完整 5-seed 训练系统；真实 NeoRL-2 训练、评估 rollout、聚合图表和论文模板/证据台账还未继续做。

> 现在skills已经完全就位了，重新读取就能看到结果，重新读取以继续进行全流程。如果现在的代码不能直接出结果的话，你需要给出下一步的操作

<details><summary>27 previous messages</summary>

> 我会先重新读取 `academic-research-suite` 和它现在可用的 workflow，确认 full pipeline 的实际入口是否已经补齐。然后检查当前 `code/` 是否已经能从数据到结果直跑；如果不能，我会把缺口和下一步操作落到具体命令/文件。
>
> 这次 ARS 已经完整：`ars/academic-pipeline/WORKFLOW.md` 等入口都存在。测试仍然通过，但当前 `code/` 只是 scaffold，不可能直接生成论文结果；我会读取 pipeline 和 experiment workflow 后，按它们的阶段要求判断下一步。
>
> 按 ARS 判断，当前材料处在“Stage 1/实验桥接到 Stage 2 之前”：研究问题和实施方案已有，但没有可验证实验结果，所以还不能进入论文结果写作。下一步应先把代码从 scaffold 推到“给定数据即可训练/评估/汇总”的状态。
>
> 我不会把“没有数据”当成唯一原因就停下，因为当前代码即使拿到数据也还缺数据准备/训练入口。下一步我先补“给定 NPZ 数据即可验证并生成 metadata”的数据层和脚本，这样阻塞点会从“代码不完整”推进到更明确的“需要真实 NeoRL-2 数据/训练环境”。
>
> 先写数据准备测试和接口桩，然后跑 RED。测试会用合成 NPZ，不依赖真实 NeoRL-2。
>
> <details><summary>Created 3 files</summary>
>
> - Wrote `./code/scripts/prepare_pipeline_data.py` (+28 -0)
> - Wrote `./code/src/dar_td3bc/data/pipeline_dataset.py` (+28 -0)
> - Wrote `./code/tests/test_pipeline_dataset.py` (+101 -0)
>
> </details>
>
> 数据准备测试与接口桩已添加。现在跑这组测试确认失败。
>
> RED 完成，4 个数据准备测试都失败在未实现接口。现在实现 NPZ 加载、形状验证和 metadata 写出。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./code/scripts/prepare_pipeline_data.py` (+35 -2)
> - Wrote `./code/src/dar_td3bc/data/pipeline_dataset.py` (+96 -3)
>
> </details>
>
> 数据层实现完成。现在跑数据准备测试和完整测试。
>
> 数据准备层已补齐，完整测试现在 31 个通过。接下来我会加一个 `check_next_operation.py`，让代码自己报告“现在为什么不能出结果、下一步该做什么”，避免之后靠口头判断。
>
> <details><summary>Created 3 files</summary>
>
> - Wrote `./code/scripts/check_next_operation.py` (+24 -0)
> - Wrote `./code/src/dar_td3bc/utils/readiness.py` (+20 -0)
> - Wrote `./code/tests/test_readiness.py` (+67 -0)
>
> </details>
>
> readiness 测试和接口桩已添加。现在跑 RED。
>
> RED 完成。现在实现 readiness 判断：先检查数据，再检查 metadata，再检查训练/聚合脚本是否存在，最后检查 `final_summary.csv`。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/src/dar_td3bc/utils/readiness.py` (+119 -2)
>
> </details>
>
> readiness 实现完成。现在跑对应测试和完整测试。
>
> readiness 判断已通过，完整测试 35 个通过。为了把下一步操作固化成文件，我会给脚本加 `--output` 参数并测试它写出 `NEXT_OPERATIONS.md`。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/tests/test_readiness.py` (+26 -0)
>
> </details>
>
> 输出文件测试已添加。现在先跑 RED。
>
> RED 正常：脚本还不支持 `--output`。现在实现这个参数。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/scripts/check_next_operation.py` (+9 -1)
>
> </details>
>
> `--output` 已实现。现在跑 readiness 和完整测试。
>
> 完整测试现在 36 个通过。我会生成当前项目的 `code/NEXT_OPERATIONS.md`，它会明确说明当前阻塞和下一条命令。
>
> `NEXT_OPERATIONS.md` 已生成。现在我会读一下内容确认它反映当前真实状态。
>
> 测试又生成了 `__pycache__`，我会清理生成缓存，保持工作区只留下有意义的源码/结果文件。
>
> 代码侧当前可完成的验证已经跑完，接下来我会按 verification skill 做最后核对：确认测试状态、生成文件、以及仍然阻塞论文结果的条件。
>
> 最终核对完成：测试 36 个通过，`check_next_operation` 明确报告当前不能出最终论文结果，且工作区没有 `__pycache__`。我会把状态更新后给你一个简短结论和下一步命令。

</details>

已重新读取完整 `academic-research-suite`，这次 `ars/academic-pipeline/WORKFLOW.md`、`experiment-agent`、`academic-paper` 都已经可用。按 ARS 判断，当前还不能进入论文 Results/完整初稿阶段，因为没有可验证实验结果。

当前代码不能直接出结果，原因已经写入 [NEXT_OPERATIONS.md](./code/NEXT_OPERATIONS.md)：

1. 缺 NeoRL-2 Pipeline 训练/验证数据：`data/raw/neorl2/pipeline_train.npz`、`pipeline_val.npz`
2. 缺 metadata：需要先跑数据验证
3. 缺完整训练/评估/聚合脚本：`train_behavior_policy.py`、`pretrain_encoder.py`、`train_td3bc.py`、`train_dar_td3bc.py`、`evaluate.py`、`aggregate_results.py`
4. 因此没有 `results/aggregated/final_summary.csv`

我已经补了数据准备层和 readiness 检查：

- [pipeline_dataset.py](./code/src/dar_td3bc/data/pipeline_dataset.py)
- [prepare_pipeline_data.py](./code/scripts/prepare_pipeline_data.py)
- [check_next_operation.py](./code/scripts/check_next_operation.py)

验证结果：

```text
python -m pytest tests -q
36 passed
```

下一步操作是：

```bash
cd code
python scripts/acquire_data.py --task Pipeline --output data/raw/neorl2
```

但在当前电脑上它已经确认会因缺 `gymnasium/neorl2` 或官方数据环境而失败并输出数据请求。代码开发的下一步则是继续实现 6 个训练/评估/聚合脚本，让项目在拿到数据后能跑出 `final_summary.csv`。

> 缺什么你告诉我啊，怎么下载？到哪里下载？

<details><summary>9 previous messages</summary>

> 我会核对官方 NeoRL-2 仓库和 Hugging Face 数据页，给你列清楚“缺什么、去哪里下、怎么放到当前项目里”。这类链接和数据集名称可能会变，所以我先用官方来源确认。
>
> <details><summary>Searched the web</summary>
>
> <details><summary>Searched the web</summary>
>
> - Searched the web for `NeoRL2 polixir GitHub Pipeline get_dataset`
>
> </details>
>
> </details>
>
> 我发现一个需要顺手修的代码问题：官方 NeoRL-2 `get_dataset()` 字段是 `obs/next_obs/action/reward/done`，而当前保存函数只接受 `observations/next_observations/actions/rewards`。我会先补这个映射，否则即使你下载成功也可能保存失败。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/tests/test_acquire_data_script.py` (+28 -0)
>
> </details>
>
> 映射测试已添加。现在先跑这个测试，确认它失败在当前实现上。
>
> RED 确认：保存函数不认识官方字段名。现在实现字段映射。
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./code/src/dar_td3bc/data/acquisition.py` (+23 -14)
>
> </details>
>
> 官方字段映射已实现。现在跑相关测试和完整测试。
>
> 字段映射修复已验证，完整测试现在 37 个通过。我会清理测试缓存，然后给你具体下载路径和命令。

</details>

缺两类东西：

1. **NeoRL-2 运行环境**
2. **Pipeline 数据**

官方位置：

- NeoRL-2 环境仓库：https://github.com/polixir/NeoRL2
- NeoRL-2 数据集：https://huggingface.co/datasets/polixirai/NeoRL2

官方说明里，`env.get_dataset()` 会返回训练集和验证集，字段是 `obs / next_obs / action / reward / done`。我刚把代码修了，现在会自动映射成我们项目内部用的 `observations / next_observations / actions / rewards / dones`。

**推荐下载方式**
先建一个 Python 3.10 环境，不建议用现在这个 3.12 环境跑正式实验：

```bash
conda create -n dar-td3bc python=3.10 -y
conda activate dar-td3bc
```

然后安装依赖：

```bash
pip install torch numpy scipy pandas pyyaml tqdm pytest scikit-learn datasets gymnasium
git clone https://github.com/polixir/NeoRL2.git ../NeoRL2
pip install -e ../NeoRL2
pip install -e ./code
```

然后在项目根目录运行：

```bash
cd code
python scripts/acquire_data.py --task Pipeline --output data/raw/neorl2
```

成功后应该得到：

```text
code/data/raw/neorl2/pipeline_train.npz
code/data/raw/neorl2/pipeline_val.npz
```

然后继续：

```bash
python scripts/prepare_pipeline_data.py ^
  --train data/raw/neorl2/pipeline_train.npz ^
  --val data/raw/neorl2/pipeline_val.npz ^
  --output-dir data/metadata
```

如果 `env.get_dataset()` 下载失败，可以手动去 Hugging Face 页面下载/缓存 `polixirai/NeoRL2` 数据，或者在能联网的机器上执行上面的 `acquire_data.py`，再把生成的两个 `.npz` 文件拷到：

```text
code/data/raw/neorl2/
```

我也做了验证：当前代码测试是通过的。

```text
37 passed
```

> pip install torch numpy scipy pandas pyyaml tqdm pytest scikit-learn datasets gymnasium
> git clone https://github.com/polixir/NeoRL2.git ../NeoRL2
> pip install -e ../NeoRL2
> pip install -e ./code，解释每一步

这几步是在准备一个能读取 NeoRL-2 Pipeline 数据、运行我们代码的 Python 环境。

```bash
pip install torch numpy scipy pandas pyyaml tqdm pytest scikit-learn datasets gymnasium
```

安装基础依赖：

- `torch`：PyTorch，训练 TD3+BC、DAR-TD3BC 神经网络用。
- `numpy`：数组处理，数据验证、指标计算用。
- `scipy`：后续统计检验、bootstrap、Wilcoxon 等会用。
- `pandas`：读写 CSV、汇总实验结果用。
- `pyyaml`：读取 `configs/model/dar_td3bc.yaml` 配置文件。
- `tqdm`：训练进度条。
- `pytest`：运行测试。
- `scikit-learn`：后续可能用于数据划分、标准化或统计工具。
- `datasets`：Hugging Face 数据集下载接口。
- `gymnasium`：创建 NeoRL-2 环境时用，例如 `gym.make("Pipeline")`。

```bash
git clone https://github.com/polixir/NeoRL2.git ../NeoRL2
```

把官方 NeoRL-2 环境代码下载到当前项目的上一级目录。

如果你现在在：

```text
D:\Study_Data\计算机控制系统\Paper
```

那它会下载到：

```text
D:\Study_Data\计算机控制系统\NeoRL2
```

这样做是为了不把第三方官方仓库塞进我们自己的 `Paper/code` 目录里。

```bash
pip install -e ../NeoRL2
```

把刚下载的 NeoRL-2 官方环境以“可编辑模式”安装进当前 Python 环境。

作用是让 Python 能执行：

```python
import neorl2
import gymnasium as gym

env = gym.make("Pipeline")
train_data, val_data = env.get_dataset()
```

`-e` 的意思是 editable install：如果以后官方仓库代码有改动，Python 直接读取那个目录，不需要重新打包安装。

```bash
pip install -e ./code
```

把我们自己的项目代码也以“可编辑模式”安装进当前 Python 环境。

作用是让 Python 能在任何脚本里导入：

```python
from dar_td3bc.data.acquisition import save_dataset_npz
from dar_td3bc.models.temporal_encoder import DelayEncoder
```

如果你当前目录是：

```text
D:\Study_Data\计算机控制系统\Paper
```

那么 `./code` 就是：

```text
D:\Study_Data\计算机控制系统\Paper\code
```

总结一下：

1. 第一行：装通用 Python/深度学习/数据处理依赖。
2. 第二行：下载官方 NeoRL-2。
3. 第三行：让当前环境能 `import neorl2`。
4. 第四行：让当前环境能 `import dar_td3bc`。

> 将仓库命名为CC_EX，推到github上面

<details><summary>11 previous messages</summary>

> 我会先检查当前目录是不是 Git 仓库、有哪些文件会被纳入提交，以及 GitHub 远程/认证是否可用。然后按 GitHub 发布流程处理；如果没有 GitHub CLI 登录或没有仓库权限，会明确告诉你卡在哪里。
>
> 当前有两个阻塞：这个目录还不是 Git 仓库，而且本机没有安装 GitHub CLI `gh`。我会检查可用的 GitHub 插件/连接器是否能直接创建仓库；如果不能，就只能先把本地仓库初始化好，并给出你需要安装/登录的步骤。
>
> GitHub 连接器目前只暴露了“操作已有仓库内容/提交/分支”的 API，没有创建新仓库的工具；本机也没有 `gh`，所以我不能直接替你新建 `CC_EX` 并 push。先把本地 Git 仓库整理成可发布状态，然后你安装/登录 `gh` 后一条命令就能创建远程仓库并推上去。
>
> 我会把 `.pytest_cache` 这类生成物排除掉，初始化本地 Git 仓库并提交当前项目。远程 GitHub 创建和 push 需要 `gh` 或你提供一个已存在的远程仓库 URL。
>
> <details><summary>Created a file</summary>
>
> - Wrote `./.gitignore` (+28 -0)
>
> </details>
>
> `.gitignore` 已加入，排除了缓存、原始数据、训练运行结果和模型 checkpoint。现在清理已有 pytest 缓存并跑一次测试，确保提交前状态可验证。
>
> 测试通过：37 个测试全部通过。现在初始化本地 Git 仓库并查看将要提交的文件，确认没有大数据或缓存被纳入。
>
> 并行执行时 `git status` 抢在 `git init` 完成前跑了，所以报了旧状态。仓库已经初始化成功，我现在重新检查状态。
>
> 当前待提交内容就是方案文档、`code/`、`docs/` 和 `.gitignore`，没有数据目录或缓存。按你“推到 GitHub”的要求，我会提交整个当前项目。
>
> 已 stage 全部项目文件，只有换行符提示，不影响提交。现在创建初始提交。
>
> 本地仓库和初始提交已经完成。现在最后确认 Git 状态和提交哈希；远程创建/推送还缺 GitHub CLI 或远程 URL。

</details>

我已经把当前项目初始化成了本地 Git 仓库，并提交好了：

```text
branch: main
commit: c65e1f1 Initial DAR-TD3BC research scaffold
```

但还没法直接推到 GitHub，因为当前机器缺两样东西：

1. 没有安装 GitHub CLI：`gh`
2. 当前仓库没有远程地址：`origin`

你有两种做法。

**推荐做法：安装 gh 后一条命令创建并推送**

在 PowerShell 里执行：

```powershell
winget install --id GitHub.cli
```

安装后重新打开 PowerShell，然后登录：

```powershell
gh auth login
```

按提示选择：

- GitHub.com
- HTTPS
- Login with a web browser

登录完成后，在当前项目目录执行：

```powershell
cd "D:\Study_Data\计算机控制系统\Paper"
gh repo create CC_EX --private --source . --remote origin --push
```

如果你想公开仓库，把 `--private` 改成：

```powershell
--public
```

**如果你已经在 GitHub 网页上创建了空仓库 `CC_EX`**

把远程地址发我，格式类似：

```text
https://github.com/你的用户名/CC_EX.git
```

或者你自己执行：

```powershell
cd "D:\Study_Data\计算机控制系统\Paper"
git remote add origin https://github.com/你的用户名/CC_EX.git
git push -u origin main
```

当前本地仓库已经准备好，差的只是 GitHub 远程创建/认证这一步。

> ？用git推送就行了啊

<details><summary>1 previous message</summary>

> 我可以用纯 `git push`，但 Git 仍然需要一个远程地址。当前本地仓库没有 `origin`，所以我先确认一下远程和 Git 配置；如果能从配置推断用户名，我再给出最短命令。

</details>

可以，纯 `git` 推送没问题。但当前仓库没有远程：

```text
git remote -v
# 空
```

所以 `git push` 现在不知道要推到哪个 GitHub 仓库。你需要先在 GitHub 网页上创建一个空仓库 `CC_EX`，然后在本地执行：

```powershell
cd "D:\Study_Data\计算机控制系统\Paper"
git remote add origin https://github.com/你的GitHub用户名/CC_EX.git
git push -u origin main
```

如果你用 SSH：

```powershell
git remote add origin git@github.com:你的GitHub用户名/CC_EX.git
git push -u origin main
```

你把 GitHub 上新建的 `CC_EX` 仓库地址发我，我就可以直接替你执行：

```powershell
git remote add origin <你的仓库地址>
git push -u origin main
```
