import numpy as np

from segagent.evaluation.evaluator import assign_lesion_size_groups


def test_assign_lesion_size_groups_includes_empty_and_tertiles() -> None:
    rows = [
        {"target_volume_mm3": value}
        for value in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    ]
    thresholds = assign_lesion_size_groups(rows)
    assert thresholds == {"small_max_mm3": 2.6666666666666665, "medium_max_mm3": 4.333333333333333}
    assert [row["lesion_size_group"] for row in rows] == [
        "empty",
        "small",
        "small",
        "medium",
        "medium",
        "large",
        "large",
    ]
