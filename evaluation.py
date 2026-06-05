from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import DATE_COL, TARGET_COL


@dataclass(frozen=True)
class EvalResult:
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier: float
    precision_at_1pct: float
    precision_at_5pct: float
    precision_at_10pct: float
    recall_at_1pct: float
    recall_at_5pct: float
    recall_at_10pct: float
    top_decile_lift: float
    base_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_clip_probs(y_prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)


def precision_recall_at_fraction(y_true: np.ndarray, y_prob: np.ndarray, fraction: float) -> tuple[float, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(y_prob)[::-1][:k]
    y_top = y_true[order]
    precision = float(y_top.mean()) if len(y_top) else 0.0
    total_positives = max(int(y_true.sum()), 1)
    recall = float(y_top.sum() / total_positives)
    return precision, recall


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    base_rate = float(np.mean(y_true))
    if base_rate == 0:
        return 0.0
    p10, _ = precision_recall_at_fraction(y_true, y_prob, 0.10)
    return float(p10 / base_rate)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> EvalResult:
    y_true = np.asarray(y_true).astype(int)
    y_prob = _safe_clip_probs(y_prob)

    p1, r1 = precision_recall_at_fraction(y_true, y_prob, 0.01)
    p5, r5 = precision_recall_at_fraction(y_true, y_prob, 0.05)
    p10, r10 = precision_recall_at_fraction(y_true, y_prob, 0.10)

    return EvalResult(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        log_loss=float(log_loss(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        precision_at_1pct=p1,
        precision_at_5pct=p5,
        precision_at_10pct=p10,
        recall_at_1pct=r1,
        recall_at_5pct=r5,
        recall_at_10pct=r10,
        top_decile_lift=float(top_decile_lift(y_true, y_prob)),
        base_rate=float(y_true.mean()),
    )


def monthly_backtest(df_eval: pd.DataFrame, prob_col: str = "prediction") -> pd.DataFrame:
    rows = []
    monthly = df_eval.copy()
    monthly["YEAR_MONTH"] = monthly[DATE_COL].dt.to_period("M").astype(str)

    for month, grp in monthly.groupby("YEAR_MONTH", sort=True):
        if grp[TARGET_COL].nunique() < 2:
            continue
        metrics = evaluate_binary(grp[TARGET_COL].to_numpy(), grp[prob_col].to_numpy()).to_dict()
        metrics["year_month"] = month
        metrics["n_rows"] = int(len(grp))
        rows.append(metrics)

    return pd.DataFrame(rows)
