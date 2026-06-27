from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_post_resume_runner_skips_completed_rollout_and_splats_seed_arrays():
    script = (PROJECT_ROOT / "run.ps1").read_text(encoding="utf-8")

    assert "evaluate_rollout.py" not in script
    assert "$td3bcRobustnessArgs" in script
    assert "$darRobustnessArgs" in script
    assert "Invoke-Python scripts/run_robustness.py `" not in script
