from segagent.data.splits import SeriesRecord, create_patient_split, validate_patient_split


def test_patient_split_is_deterministic_exact_and_leak_free() -> None:
    records = [
        SeriesRecord(f"patient-{index:02d}", f"series-{index:02d}", index % 6)
        for index in range(20)
    ]
    records.append(SeriesRecord("patient-00", "series-extra", 1))
    ratios = {"train": 0.7, "validation": 0.15, "test": 0.15}

    first = create_patient_split(records, ratios, seed=42)
    second = create_patient_split(reversed(records), ratios, seed=42)

    assert first == second
    assert first["patient_counts"] == {"train": 14, "validation": 3, "test": 3}
    validate_patient_split(first, records)
    patient_zero_split = next(
        name for name, patients in first["patients"].items() if "patient-00" in patients
    )
    assert {"series-00", "series-extra"}.issubset(first["series"][patient_zero_split])
