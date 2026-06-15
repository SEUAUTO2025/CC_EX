import csv

from dar_td3bc.utils.provenance import (
    official_pipeline_baselines,
    render_environment_txt,
    write_reported_baselines_csv,
)


def test_official_pipeline_baselines_are_marked_reported_not_rerun():
    rows = official_pipeline_baselines(retrieved_date="2026-06-15")

    assert len(rows) == 10
    td3bc = next(row for row in rows if row["method"] == "TD3+BC")
    assert td3bc["mean"] == "81.95"
    assert td3bc["provenance"] == "reported_not_rerun"
    assert td3bc["error_type"] == "unverified_error_bar"
    assert "verify" in td3bc["notes"].lower()


def test_write_reported_baselines_csv(tmp_path):
    output = write_reported_baselines_csv(tmp_path / "reported.csv")

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 10
    assert rows[0]["provenance"] == "reported_not_rerun"


def test_render_environment_txt_is_a_planning_record_not_local_claim():
    text = render_environment_txt()

    assert "planning record only" in text
    assert "Python: 3.10" in text
    assert "do not need to" in text
