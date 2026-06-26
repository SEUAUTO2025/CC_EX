import sys

from dar_td3bc.utils.parallel import run_parallel_commands


def test_run_parallel_commands_returns_zero_when_all_commands_succeed(tmp_path):
    output_a = tmp_path / "a.txt"
    output_b = tmp_path / "b.txt"
    commands = [
        [sys.executable, "-c", f"from pathlib import Path; Path(r'{output_a}').write_text('a')"],
        [sys.executable, "-c", f"from pathlib import Path; Path(r'{output_b}').write_text('b')"],
    ]

    code = run_parallel_commands(commands, max_workers=2, label="test")

    assert code == 0
    assert output_a.read_text() == "a"
    assert output_b.read_text() == "b"


def test_run_parallel_commands_returns_failing_exit_code():
    commands = [
        [sys.executable, "-c", "raise SystemExit(0)"],
        [sys.executable, "-c", "raise SystemExit(7)"],
    ]

    code = run_parallel_commands(commands, max_workers=2, label="test")

    assert code == 7
