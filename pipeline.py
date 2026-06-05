from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import (
    DATE_COL,
    DEFAULT_FEATURES,
    ID_COL,
    TARGET_COL,
    TrainingConfig,
    default_model_params,
)
from .data import (
    FeatureSchema,
    apply_feature_schema,
    build_feature_schema,
    estimate_scale_pos_weight,
    load_dataset,
    slice_snapshot,
)
from .evaluation import evaluate_binary, monthly_backtest
from .io_utils import ensure_dir, save_dataframe, save_joblib, save_json
from .modeling import apply_calibrator, fit_calibrator, train_lightgbm
from .splits import build_time_folds, materialize_fold


def run_training(config: TrainingConfig) -> dict:
    artifacts_dir = ensure_dir(config.artifacts_dir)
    df = load_dataset(config.data_path)
    schema = build_feature_schema(df, DEFAULT_FEATURES)
    df = apply_feature_schema(df, schema)

    folds = build_time_folds(
        df=df,
        n_folds=config.n_folds,
        train_window_days=config.train_window_days,
        val_window_days=config.val_window_days,
        gap_days=config.gap_days,
    )

    fold_metrics = []
    all_val_preds = []
    latest_booster = None
    latest_calibrator = None

    for fold in folds:
        train_df, val_df = materialize_fold(df, fold)

        X_train = train_df[schema.features]
        y_train = train_df[TARGET_COL]
        X_val = val_df[schema.features]
        y_val = val_df[TARGET_COL]

        spw = estimate_scale_pos_weight(y_train)
        model_params = config.model_params or default_model_params(scale_pos_weight=spw)

        booster = train_lightgbm(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_params=model_params,
            num_boost_round=config.num_boost_round,
            early_stopping_rounds=config.early_stopping_rounds,
        )

        raw_val = booster.predict(X_val, num_iteration=booster.best_iteration)
        calibrator = fit_calibrator(raw_val, y_val.to_numpy(), method=config.calibration_method)
        cal_val = apply_calibrator(calibrator, raw_val)

        metrics = evaluate_binary(y_val.to_numpy(), cal_val).to_dict()
        metrics.update(
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "val_start": fold.val_start,
                "val_end": fold.val_end,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "scale_pos_weight": float(spw),
                "best_iteration": int(booster.best_iteration),
            }
        )
        fold_metrics.append(metrics)

        fold_pred_df = val_df[[DATE_COL, ID_COL, TARGET_COL]].copy()
        fold_pred_df["prediction"] = cal_val
        fold_pred_df["raw_prediction"] = raw_val
        fold_pred_df["fold_id"] = fold.fold_id
        all_val_preds.append(fold_pred_df)

        latest_booster = booster
        latest_calibrator = calibrator

    if latest_booster is None:
        raise RuntimeError("No model was trained.")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    val_pred_df = pd.concat(all_val_preds, axis=0, ignore_index=True)
    monthly_df = monthly_backtest(val_pred_df, prob_col="prediction")

    metrics_summary = {
        "fold_mean": fold_metrics_df.mean(numeric_only=True).to_dict(),
        "fold_std": fold_metrics_df.std(numeric_only=True).to_dict(),
        "n_validation_rows": int(len(val_pred_df)),
    }

    latest_booster.save_model(str(Path(artifacts_dir) / "model.txt"))
    save_joblib(latest_calibrator, Path(artifacts_dir) / "calibrator.joblib")
    save_json(schema.to_dict(), Path(artifacts_dir) / "feature_schema.json")
    save_json(config.to_dict(), Path(artifacts_dir) / "training_config.json")
    save_json(metrics_summary, Path(artifacts_dir) / "metrics_summary.json")
    save_dataframe(fold_metrics_df, Path(artifacts_dir) / "fold_metrics.csv")
    save_dataframe(monthly_df, Path(artifacts_dir) / "backtest_monthly.csv")
    save_dataframe(val_pred_df, Path(artifacts_dir) / "validation_predictions.parquet")

    return {
        "artifacts_dir": str(artifacts_dir),
        "metrics_summary": metrics_summary,
        "fold_metrics": fold_metrics_df,
        "backtest_monthly": monthly_df,
    }


def run_scoring(
    data_path: str,
    artifacts_dir: str,
    output_path: str,
    snapshot_date: str | None = None,
) -> pd.DataFrame:
    from .io_utils import load_joblib, load_json
    import lightgbm as lgb

    df = load_dataset(data_path)
    schema = FeatureSchema.from_dict(load_json(Path(artifacts_dir) / "feature_schema.json"))
    df = apply_feature_schema(df, schema)

    score_df = slice_snapshot(df, snapshot_date=snapshot_date)
    booster = lgb.Booster(model_file=str(Path(artifacts_dir) / "model.txt"))
    calibrator = load_joblib(Path(artifacts_dir) / "calibrator.joblib")

    raw_scores = booster.predict(score_df[schema.features], num_iteration=booster.best_iteration)
    calibrated_scores = apply_calibrator(calibrator, raw_scores)

    output = score_df[[DATE_COL, ID_COL]].copy()
    output["churn_probability"] = calibrated_scores
    output["raw_score"] = raw_scores
    output["rank_desc"] = output["churn_probability"].rank(method="first", ascending=False).astype(int)
    output = output.sort_values(["rank_desc", ID_COL]).reset_index(drop=True)

    save_dataframe(output, output_path)
    return output
