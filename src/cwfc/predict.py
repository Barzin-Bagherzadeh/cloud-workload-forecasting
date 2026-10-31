import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def build_target_dataframe(data_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path / "vm_cpu_readings.csv.gz")

    # Keep only valid CPU observations
    data = data.dropna(subset=["timestamp", "vm_id", "avg_cpu"])

    groupby_time = data.groupby("timestamp")

    unique_vm_id = groupby_time["vm_id"].nunique()
    mean_avg_cpu = groupby_time["avg_cpu"].mean()

    final_dataframe = pd.DataFrame({
        "active_vm_count": unique_vm_id,
        "aggregate_cpu_utilization": mean_avg_cpu
    })

    final_dataframe = final_dataframe.reset_index()
    final_dataframe = final_dataframe.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return final_dataframe


def split_data(
    final_dataframe: pd.DataFrame,
    test_size: int = 24
):
    if test_size <= 0:
        raise ValueError("test_size must be positive")

    if len(final_dataframe) <= test_size:
        raise ValueError("Not enough observations for training and testing")

    learning_dataframe = final_dataframe.iloc[:-test_size].copy()
    test_dataframe = final_dataframe.iloc[-test_size:].copy()

    return learning_dataframe, test_dataframe


def add_median_feature(learning_dataframe: pd.DataFrame) -> pd.DataFrame:
    new_dataframe = pd.DataFrame({
        "timestamp": learning_dataframe["timestamp"],
        "aggregate_cpu_utilization": (
            learning_dataframe["aggregate_cpu_utilization"]
        )
    })

    data = new_dataframe["aggregate_cpu_utilization"]

    median_values = []
    median_values.append(9.256428)

    for i in range(1, len(new_dataframe)):
        new_median = data.iloc[max(0, i - 4):i].median()
        median_values.append(new_median)

    new_dataframe["median_4"] = median_values

    return new_dataframe


def train_model(new_dataframe: pd.DataFrame):
    train_final = new_dataframe.copy()

    model_final = SGDRegressor(
        max_iter=1000,
        random_state=42
    )

    scaler_final = StandardScaler()

    train_array_scaled_final = scaler_final.fit_transform(
        train_final[["timestamp", "median_4"]]
    )

    model_final.fit(
        train_array_scaled_final,
        train_final["aggregate_cpu_utilization"]
    )

    return model_final, scaler_final


def forecast_cpu(
    model_final,
    scaler_final,
    train_final: pd.DataFrame,
    test_dataframe: pd.DataFrame
) -> list[float]:

    # Only timestamps are needed from the test dataframe.
    new_df_final = pd.DataFrame({
        "timestamp": test_dataframe["timestamp"]
    })

    history_final = (
        train_final["aggregate_cpu_utilization"].tail(4).tolist()
    )

    predictions_final = []

    for index in new_df_final.index:
        median_value_final = np.median(history_final[-4:])

        current_features_final = pd.DataFrame({
            "timestamp": [new_df_final.loc[index, "timestamp"]],
            "median_4": [median_value_final]
        })

        current_features_scaled_final = scaler_final.transform(
            current_features_final
        )

        prediction_final = model_final.predict(
            current_features_scaled_final
        )[0]

        # Keep predictions within the valid CPU range.
        prediction_final = float(np.clip(prediction_final, 0, 100))

        predictions_final.append(prediction_final)
        history_final.append(prediction_final)

    return predictions_final


def build_predictions(
    test_dataframe: pd.DataFrame,
    cpu_predictions: list[float],
    learning_dataframe: pd.DataFrame
) -> pd.DataFrame:

    if learning_dataframe["active_vm_count"].nunique() == 1:
        active_vm_prediction = learning_dataframe["active_vm_count"].iloc[0]
    else:
        active_vm_prediction = learning_dataframe["active_vm_count"].iloc[-1]

    predictions = pd.DataFrame({
        "timestamp": test_dataframe["timestamp"].to_numpy(),
        "active_vm_count": active_vm_prediction,
        "aggregate_cpu_utilization": cpu_predictions
    })

    return predictions


def validate_predictions(predictions: pd.DataFrame) -> None:
    expected_columns = [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization"
    ]

    if list(predictions.columns) != expected_columns:
        raise ValueError("Invalid prediction columns")

    if len(predictions) != 24:
        raise ValueError("Predictions must contain exactly 24 rows")

    if predictions.isnull().any().any():
        raise ValueError("Predictions contain missing values")

    if not predictions["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted")

    if predictions["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found")

    if not np.isfinite(
        predictions.select_dtypes(include="number")
    ).all().all():
        raise ValueError("Predictions contain non-finite values")

    if (predictions["active_vm_count"] < 0).any():
        raise ValueError("Active VM count cannot be negative")

    if not predictions["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("CPU utilization must be between 0 and 100")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the challenge_data directory"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output predictions.csv file"
    )

    args = parser.parse_args()

    final_dataframe = build_target_dataframe(args.data)

    learning_dataframe, test_dataframe = split_data(final_dataframe)

    new_dataframe = add_median_feature(learning_dataframe)

    model_final, scaler_final = train_model(new_dataframe)

    predictions_final = forecast_cpu(
        model_final,
        scaler_final,
        new_dataframe,
        test_dataframe
    )

    predictions = build_predictions(
        test_dataframe,
        predictions_final,
        learning_dataframe
    )

    validate_predictions(predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(args.output, index=False)

    print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()