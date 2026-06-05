from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import (
    DATE_COL,
    DEFAULT_FEATURES,
    DEFAULT_FILL_VALUES,
    ID_COL,
    TARGET_COL,
)


@dataclass(frozen=True)
class FeatureSchema:
    features: list[str]
    fill_values: dict[str, float]

    def to_dict(self) -> dict:
        return {"features": self.features, "fill_values": self.fill_values}

    @classmethod
    def from_dict(cls, payload: dict) -> "FeatureSchema":
        return cls(features=list(payload["features"]), fill_values=dict(payload["fill_values"]))


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if DATE_COL not in df.columns or ID_COL not in df.columns:
        raise ValueError(f"Dataset must contain {DATE_COL} and {ID_COL}.")
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    return df


def validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_feature_schema(df: pd.DataFrame, features: Sequence[str] | None = None) -> FeatureSchema:
    features = list(features or DEFAULT_FEATURES)
    validate_required_columns(df, [DATE_COL, ID_COL, TARGET_COL, *features])

    fill_values = {}
    for feature in features:
        fill_values[feature] = DEFAULT_FILL_VALUES.get(feature, 0.0)
    return FeatureSchema(features=features, fill_values=fill_values)


def apply_feature_schema(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    validate_required_columns(df, [DATE_COL, ID_COL, *schema.features])

    out = df.copy()
    for col, val in schema.fill_values.items():
        out[col] = out[col].fillna(val)

    for col in schema.features:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(schema.fill_values.get(col, 0.0))

    out = out.sort_values([DATE_COL, ID_COL]).reset_index(drop=True)
    return out


def latest_snapshot(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df[DATE_COL].max())


def slice_snapshot(df: pd.DataFrame, snapshot_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    date = pd.Timestamp(snapshot_date) if snapshot_date is not None else latest_snapshot(df)
    return df.loc[df[DATE_COL] == date].copy()


def estimate_scale_pos_weight(y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y)
    pos = int((y_arr == 1).sum())
    neg = int((y_arr == 0).sum())
    if pos == 0:
        return 1.0
    return neg / pos
