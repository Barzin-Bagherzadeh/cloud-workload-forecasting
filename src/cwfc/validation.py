from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: int
    validation_start: int
    validation_end: int


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 20,
    min_train_size: int = 60,
    step: int = 10,
) -> list[TimeSplit]:
    """Return expanding-window splits for evenly sampled time series.

    horizon is the number of observations in the validation period.
    """
    if horizon <= 0 or min_train_size <= 0 or step <= 0:
        raise ValueError("horizon,min_train_size, and step must be positive")
    ts = frame[timestamp_col]
    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted")
    n = len(frame)
    out: list[TimeSplit] = []
    train_end_idx = min_train_size
    while train_end_idx + horizon <= n:
        train_end = ts.iloc[train_end_idx - 1]
        val_start = ts.iloc[train_end_idx]
        val_end = ts.iloc[train_end_idx + horizon - 1]
        out.append(TimeSplit(train_end, val_start, val_end))
        train_end_idx += step
    return out