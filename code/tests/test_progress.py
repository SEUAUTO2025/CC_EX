from dar_td3bc.utils.progress import progress_iter, progress_range


def test_progress_range_yields_one_based_steps_when_disabled():
    assert list(progress_range(3, desc="train", enabled=False)) == [1, 2, 3]


def test_progress_iter_yields_items_when_disabled():
    assert list(progress_iter(["a", "b"], desc="rollout", enabled=False)) == [
        "a",
        "b",
    ]
