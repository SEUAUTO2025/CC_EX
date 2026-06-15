import csv

from dar_td3bc.utils.run import append_csv_row, load_yaml, make_run_dir


def test_load_yaml_reads_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  batch_size: 16\n", encoding="utf-8")

    config = load_yaml(path)

    assert config["train"]["batch_size"] == 16


def test_make_run_dir_uses_method_seed_and_run_name(tmp_path):
    run_dir = make_run_dir(tmp_path, "td3bc", 3, run_name="smoke")

    assert run_dir.exists()
    assert run_dir.name == "smoke_seed3"


def test_append_csv_row_writes_header_once(tmp_path):
    path = tmp_path / "metrics.csv"
    append_csv_row(path, {"step": 1, "loss": 2.0})
    append_csv_row(path, {"step": 2, "loss": 1.0})

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [{"step": "1", "loss": "2.0"}, {"step": "2", "loss": "1.0"}]
