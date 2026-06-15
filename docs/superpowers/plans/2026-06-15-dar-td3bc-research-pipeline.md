# DAR-TD3BC Research Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable `code/` research scaffold for DAR-TD3BC and a provenance-safe paper scaffold that can later be filled only from verified experiment outputs.

**Architecture:** The first pass implements core offline-RL utilities that can be tested without NeoRL-2 installed: observation parsing, trajectory indexing, control metrics, temporal models, and provenance scaffolding. Environment/data acquisition scripts fail explicitly when dependencies or data are missing instead of fabricating results.

**Tech Stack:** Python 3.10 target, current local Python 3.12 smoke tests, PyTorch, NumPy, Pandas, PyYAML, pytest, SAGE LaTeX template for manuscript work.

---

### Task 1: Project Skeleton and RED Tests

**Files:**
- Create: `code/pyproject.toml`
- Create: `code/src/dar_td3bc/__init__.py`
- Create: `code/src/dar_td3bc/data/pipeline_obs.py`
- Create: `code/src/dar_td3bc/data/trajectory_index.py`
- Create: `code/src/dar_td3bc/evaluation/control_metrics.py`
- Create: `code/src/dar_td3bc/models/temporal_encoder.py`
- Create: `code/tests/test_obs_parser.py`
- Create: `code/tests/test_trajectory_index.py`
- Create: `code/tests/test_control_metrics.py`
- Create: `code/tests/test_temporal_encoder.py`

- [ ] **Step 1: Write failing tests and interface stubs**

Create tests that assert exact behavior for 52-dimensional Pipeline observations, episode-safe future-flow targets, control metrics, and causal encoder output shape.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest code/tests -q`
Expected: FAIL because interface stubs raise `NotImplementedError`.

### Task 2: Core Data and Metric Implementations

**Files:**
- Modify: `code/src/dar_td3bc/data/pipeline_obs.py`
- Modify: `code/src/dar_td3bc/data/trajectory_index.py`
- Modify: `code/src/dar_td3bc/evaluation/control_metrics.py`

- [ ] **Step 1: Implement observation parser**

Implement `split_pipeline_obs` so it validates `[B, 52]`, returns current flow, target flow, flat observation, and `[B, 25, 2]` history ordered oldest-to-newest.

- [ ] **Step 2: Implement trajectory-safe future targets**

Implement `TrajectoryIndex.future_flow_targets` so horizons crossing episode boundaries receive mask `0` and do not copy across trajectories.

- [ ] **Step 3: Implement control metrics**

Implement RMSE, MAE, IAE, action total variation, action energy, and target-switch settling events with censored episodes.

- [ ] **Step 4: Run tests to verify GREEN for data/metrics**

Run: `python -m pytest code/tests/test_obs_parser.py code/tests/test_trajectory_index.py code/tests/test_control_metrics.py -q`
Expected: PASS.

### Task 3: Model Building Blocks

**Files:**
- Modify: `code/src/dar_td3bc/models/temporal_encoder.py`
- Create: `code/src/dar_td3bc/models/policies.py`
- Create: `code/src/dar_td3bc/models/critic.py`
- Create: `code/tests/test_models.py`

- [ ] **Step 1: Add model tests first**

Test causal convolution length preservation, encoder latent shape, policy action range, critic output shape, prediction-head horizon count, and gate range.

- [ ] **Step 2: Run model tests to verify RED**

Run: `python -m pytest code/tests/test_temporal_encoder.py code/tests/test_models.py -q`
Expected: FAIL on missing implementations.

- [ ] **Step 3: Implement minimal PyTorch modules**

Implement `DelayEncoder`, `BehaviorPolicy`, `ResidualActor`, `TwinCritic`, `MultiHorizonPredictionHead`, and `critic_disagreement_gate`.

- [ ] **Step 4: Run model tests to verify GREEN**

Run: `python -m pytest code/tests/test_temporal_encoder.py code/tests/test_models.py -q`
Expected: PASS.

### Task 4: Scripts, Configs, and Provenance

**Files:**
- Create: `code/configs/model/dar_td3bc.yaml`
- Create: `code/scripts/inspect_environment.py`
- Create: `code/scripts/import_reported_baselines.py`
- Create: `code/src/dar_td3bc/utils/provenance.py`
- Create: `code/tests/test_provenance.py`

- [ ] **Step 1: Write provenance tests first**

Test environment report JSON includes Python, OS, PyTorch availability, and ARS degradation note.

- [ ] **Step 2: Implement scripts**

Implement scripts that create `results/environment_report.json` and `results/reported/neorl2_pipeline_official.csv` with provenance `reported_not_rerun`.

- [ ] **Step 3: Run targeted tests**

Run: `python -m pytest code/tests/test_provenance.py -q`
Expected: PASS.

### Task 5: Paper Scaffold and ARS Gates

**Files:**
- Create: `paper/evidence/source_ledger.csv`
- Create: `paper/journal_check.md`
- Create: `paper/novelty_audit.md`
- Create: `paper/unresolved_items.md`
- Create: `paper/template_check.md`
- Create: `paper/manuscript/main.tex`
- Create: `paper/manuscript/references.bib`
- Create: `paper/supplementary/hyperparameters.md`

- [ ] **Step 1: Verify current TIMC/SAGE/template facts**

Use official SAGE/TIMC sources and record checked date and source URLs. Mark Clarivate JCR items as requiring manual subscription verification if inaccessible.

- [ ] **Step 2: Download and unpack SAGE template**

Save the zip under `paper/template/`; if download fails, record the failure and do not pretend the template was verified.

- [ ] **Step 3: Create placeholder-safe manuscript scaffold**

Use the SAGE template only if downloaded; otherwise create `unresolved_items.md` and postpone manuscript正文 generation. Any abstract/results placeholders must be explicit.

### Task 6: Verification

**Files:**
- Modify as needed only to make tests pass or document blockers.

- [ ] **Step 1: Run full local test suite**

Run: `python -m pytest code/tests -q`
Expected: PASS.

- [ ] **Step 2: Run environment/provenance script**

Run: `python code/scripts/inspect_environment.py --output results/environment_report.json`
Expected: JSON report is written.

- [ ] **Step 3: Report unresolved blockers**

List missing NeoRL-2 data, unavailable Clarivate/JCR subscription checks, unavailable LaTeX compiler, and any tests that could not run.
