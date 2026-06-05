from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

import pandas as pd

from .config import DATE_COL


@dataclass(frozen=True)
class TimeFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


def build_time_folds(
    df: pd.DataFrame,
    n_folds: int,
    train_window_days: int,
    val_window_days: int,
    gap_days: int,
) -> List[TimeFold]:
    if df.empty:
        raise ValueError("Empty dataframe passed to build_time_folds().")

    max_date = pd.Timestamp(df[DATE_COL].max()).normalize()
    folds: list[TimeFold] = []

    for i in range(n_folds, 0, -1):
        val_end = max_date - pd.Timedelta(days=(i - 1) * val_window_days)
        val_start = val_end - pd.Timedelta(days=val_window_days - 1)
        train_end = val_start - pd.Timedelta(days=gap_days)
        train_start = train_end - pd.Timedelta(days=train_window_days - 1)

        folds.append(
            TimeFold(
                fold_id=n_folds - i + 1,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
        )

    valid_folds = [f for f in folds if f.train_start <= f.train_end and f.val_start <= f.val_end]
    if len(valid_folds) != len(folds):
        raise ValueError("Invalid fold configuration. Reduce fold count or window sizes.")
    return valid_folds


def materialize_fold(df: pd.DataFrame, fold: TimeFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = (df[DATE_COL] >= fold.train_start) & (df[DATE_COL] <= fold.train_end)
    val_mask = (df[DATE_COL] >= fold.val_start) & (df[DATE_COL] <= fold.val_end)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Fold {fold.fold_id} is empty. "
            f"Train rows={len(train_df)}, val rows={len(val_df)}. "
            f"Check date coverage and window configuration."
        )
    return train_df, val_df
