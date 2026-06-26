---
name: rl-training-code-quality
description: Use when writing or modifying reinforcement learning or machine learning training code, especially CUDA/GPU training, offline replay-buffer sampling, long-running optimization loops, rollout evaluation, ablation studies, robustness studies, or multi-seed experiment scripts. Enforces progress reporting, GPU-aware data loading, and parallel seed execution by default.
---

# RL Training Code Quality

## Overview

Use this skill to make training code observable, GPU-efficient, and ready for multi-seed experiments. The default implementation should expose visible progress, avoid CPU-bound data sampling bottlenecks, and run independent seeds concurrently unless the user explicitly asks for serial execution.

## Required Defaults

### Progress Reporting

- Add progress bars to every long optimization, evaluation, data conversion, ablation, robustness, and rollout loop.
- Prefer the project progress helper when one exists; otherwise use `tqdm` with a stable description and total.
- Print stage start and finish messages including method, seed, device, batch size, number of steps, and output directory.
- Keep progress updates flushed so they remain visible from PowerShell, terminals, and redirected logs.

### GPU And Data Loading

- Default to CUDA when available or when the caller passes `--device cuda`; keep an explicit CPU fallback.
- Avoid Python per-sample collation in replay-buffer or offline RL sampling loops.
- Prefer preloaded contiguous tensors, GPU-resident batch sampling, vectorized index selection, pinned memory, and non-blocking host-to-device transfer where appropriate.
- Do not increase batch size blindly. First check whether the bottleneck is CPU sampling, environment rollout, synchronization, or model compute.
- Log the resolved device and effective batch size before training begins.

### Multi-Seed Execution

- Every training entry point should support both `--seed` and `--seeds`.
- `--seed` remains the single-run compatibility path.
- `--seeds 0 1 2 3 4` should launch independent seed runs concurrently by default.
- Expose `--max-workers`; default to the number of seeds when not specified.
- Use isolated subprocesses for independent seeds so failures and RNG state remain isolated.
- Ensure run directories include the seed, for example `<run_name>_seed0`, so parallel jobs cannot overwrite each other.
- If any seed fails, return a non-zero exit code and stop downstream experiment stages.

## Experiment Script Pattern

Use this pattern for full experiment drivers:

1. Validate or acquire data.
2. Write dataset metadata and environment records.
3. Train all baseline seeds in parallel.
4. Train all proposed-method seeds in parallel.
5. Evaluate completed checkpoints.
6. Run ablations with seeds parallelized inside each variant.
7. Run robustness evaluation.
8. Aggregate results only after all upstream commands succeed.
9. Write a final readiness or next-operation report.

## Verification Checklist

- Add tests that script `--help` exposes `--seeds` and `--max-workers`.
- Add tests for the parallel command runner covering all-success and one-failure cases.
- Add tests for batch sampler output shape, dtype, and device behavior.
- Smoke-test training on a tiny step count when data is available.
- Parse-check shell or PowerShell experiment scripts before committing.
- Run the relevant test suite before claiming completion.
