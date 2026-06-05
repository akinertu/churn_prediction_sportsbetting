from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_joblib(obj: Any, path: str | Path) -> None:
    joblib.dump(obj, path)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(path)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe output format: {path.suffix}")
