# DAR-TD3BC Research Context

This context defines the experiment language for the DAR-TD3BC Pipeline study. It exists to prevent smoke checks, offline validation diagnostics, reported baselines, and paper-grade results from being mixed.

## Language

**Pipeline Task**:
The NeoRL-2 delayed flow-tracking control task with 52-dimensional observations and one-dimensional normalized inlet actions.
_Avoid_: pipeline dataset when referring to the environment, real industrial pipeline.

**Smoke Run**:
A short run used only to verify that code executes, checkpoints save, and metrics are finite. It is not evidence for paper conclusions.
_Avoid_: final experiment, paper result.

**Offline Validation Diagnostic**:
A fixed-dataset evaluation that computes policy action agreement and logged-trajectory control descriptors from validation NPZ arrays. It does not measure closed-loop environment performance.
_Avoid_: rollout result, normalized return.

**Rollout Evaluation**:
Closed-loop evaluation in the NeoRL-2 Pipeline environment where the learned policy acts for complete episodes and records rewards, trajectories, and control metrics.
_Avoid_: validation NPZ evaluation.

**Normalized Return**:
The official NeoRL-2 score computed by the benchmark's scoring convention from rollout returns.
_Avoid_: raw return, offline action MSE.

**Internal Baseline**:
A method trained and evaluated inside this project with the same data, seeds, evaluation protocol, and aggregation rules as DAR-TD3BC.
_Avoid_: reported baseline.

**Reported Baseline**:
A literature or official benchmark value copied with provenance and not re-run in this project.
_Avoid_: internal baseline, reproduced result.

**Ablation**:
An internal experiment that removes or changes one DAR-TD3BC component while holding the common training and evaluation protocol fixed.
_Avoid_: separate baseline, robustness test.

**Robustness Test**:
An evaluation or retraining setting that changes data fraction, observation noise, delay, or target-switch behavior to test sensitivity.
_Avoid_: ablation.

**Full Paper Experiment**:
The paper-grade result package containing five training seeds, closed-loop rollout evaluation, internal baselines, ablations, robustness tests, statistical aggregation, computational cost, and provenance.
_Avoid_: smoke run, single-seed result.

## Example Dialogue

Researcher: "Can we use the validation NPZ action MSE as the main result?"

Developer: "No. That is an Offline Validation Diagnostic. The main result needs Rollout Evaluation and Normalized Return, with Internal Baselines under the same protocol."

Researcher: "Can the NeoRL-2 table be compared against our ablation statistics?"

Developer: "Only as a Reported Baseline with provenance. It cannot be used in paired tests against our Internal Baselines."
