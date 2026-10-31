import pandas as pd
import pytest

from cwfc.predict import validate_predictions


def create_valid_predictions():
    return pd.DataFrame({
        "timestamp": list(range(33000, 40200, 300)),
        "active_vm_count": [10000] * 24,
        "aggregate_cpu_utilization": [9.0] * 24
    })


def test_valid_predictions():
    predictions = create_valid_predictions()

    validate_predictions(predictions)


def test_invalid_number_of_rows():
    predictions = create_valid_predictions().iloc[:-1]

    with pytest.raises(ValueError):
        validate_predictions(predictions)


def test_duplicate_timestamps():
    predictions = create_valid_predictions()
    predictions.loc[1, "timestamp"] = predictions.loc[0, "timestamp"]

    with pytest.raises(ValueError):
        validate_predictions(predictions)


def test_cpu_out_of_range():
    predictions = create_valid_predictions()
    predictions.loc[0, "aggregate_cpu_utilization"] = 110

    with pytest.raises(ValueError):
        validate_predictions(predictions)


def test_negative_active_vm_count():
    predictions = create_valid_predictions()
    predictions.loc[0, "active_vm_count"] = -1

    with pytest.raises(ValueError):
        validate_predictions(predictions)