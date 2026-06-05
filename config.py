from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


TARGET_COL = "CHURN"
DATE_COL = "DATE"
ID_COL = "CUSTOMER_ID"

DEFAULT_FEATURES: List[str] = [
    "DURATION",
    "OFF_SEASON",
    "DAYS_SINCE_LAST_BET",
    "DAYS_SINCE_LAST_DEPOSIT",
    "DAYS_SINCE_LAST_BONUS",
    "DAYS_SINCE_LAST_WITHDRAW",
    "BET_RECENCY_FACTOR",
    "DEPOSIT_RECENCY_FACTOR",
    "ACTIVITY_RATE_14D",
    "AVG_ACTIVE_WEEK_45D",
    "AVG_BET_45D",
    "WIN_RATE_14D",
    "PAYOUT_45D",
    "AVG_BET_14_90",
    "PAYOUT_14_90",
]

# Deterministic fill values to keep training/scoring identical.
DEFAULT_FILL_VALUES: Dict[str, float] = {
    "DURATION": 0.0,
    "OFF_SEASON": 0.0,
    "DAYS_SINCE_LAST_BET": 999.0,
    "DAYS_SINCE_LAST_DEPOSIT": 999.0,
    "DAYS_SINCE_LAST_BONUS": 999.0,
    "DAYS_SINCE_LAST_WITHDRAW": 999.0,
    "BET_RECENCY_FACTOR": 0.0,
    "DEPOSIT_RECENCY_FACTOR": 0.0,
    "ACTIVITY_RATE_14D": 0.0,
    "AVG_ACTIVE_WEEK_45D": 0.0,
    "AVG_BET_45D": 0.0,
    "WIN_RATE_14D": 0.0,
    "PAYOUT_45D": 0.0,
    "AVG_BET_14_90": 0.0,
    "PAYOUT_14_90": 0.0,
}


@dataclass(frozen=True)
class TrainingConfig:
    data_path: str
    artifacts_dir: str = "artifacts"
    train_window_days: int = 365
    n_folds: int = 4
    val_window_days: int = 30
    gap_days: int = 3
    calibration_method: str = "isotonic"
    random_state: int = 42
    early_stopping_rounds: int = 100
    num_boost_round: int = 2000
    model_params: dict | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


def default_model_params(scale_pos_weight: float | None = None) -> dict:
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 100,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "num_threads": -1,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)
    return params
