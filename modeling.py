from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import default_model_params


@dataclass
class TrainedArtifacts:
    booster: lgb.Booster
    calibrator: object | None
    calibrator_method: str | None


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    model_params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    booster = lgb.train(
        params=model_params,
        train_set=dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )
    return booster


def fit_calibrator(raw_scores: np.ndarray, y_true: np.ndarray, method: str = "isotonic"):
    raw_scores = np.asarray(raw_scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_scores, y_true)
        return calibrator

    if method == "platt":
        clf = LogisticRegression(max_iter=1000)
        clf.fit(raw_scores.reshape(-1, 1), y_true)
        return clf

    if method in (None, "none"):
        return None

    raise ValueError(f"Unsupported calibration method: {method}")


def apply_calibrator(calibrator, raw_scores: np.ndarray) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=float)

    if calibrator is None:
        return raw_scores

    if hasattr(calibrator, "transform"):
        return np.asarray(calibrator.transform(raw_scores), dtype=float)

    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1], dtype=float)

    raise TypeError("Unsupported calibrator object.")
