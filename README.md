# Churn Model Production Package

Production-ready churn prediction pipeline for snapshot-based betting churn modeling.

## What it does

- Loads `pre.parquet`
- Enforces schema and basic missing-value handling
- Builds time-aware expanding-window validation folds
- Trains a LightGBM model
- Evaluates ranking, classification, and calibration quality
- Fits an isotonic calibrator on the most recent validation fold
- Saves all artifacts for daily batch scoring
- Scores the latest snapshot (or any given snapshot date)

## Expected columns

Target:
- `CHURN`

Metadata:
- `DATE`
- `CUSTOMER_ID`

Features:
- `DURATION`
- `OFF_SEASON`
- `DAYS_SINCE_LAST_BET`
- `DAYS_SINCE_LAST_DEPOSIT`
- `DAYS_SINCE_LAST_BONUS`
- `DAYS_SINCE_LAST_WITHDRAW`
- `BET_RECENCY_FACTOR`
- `DEPOSIT_RECENCY_FACTOR`
- `ACTIVITY_RATE_14D`
- `AVG_ACTIVE_WEEK_45D`
- `AVG_BET_45D`
- `WIN_RATE_14D`
- `PAYOUT_45D`
- `AVG_BET_14_90`
- `PAYOUT_14_90`

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m churn_model.train   --data-path pre.parquet   --artifacts-dir artifacts   --train-window-days 365   --n-folds 4   --val-window-days 30   --gap-days 3
```

## Daily scoring

Latest available snapshot:

```bash
python -m churn_model.score   --data-path pre.parquet   --artifacts-dir artifacts   --output-path scores_latest.parquet
```

Specific snapshot date:

```bash
python -m churn_model.score   --data-path pre.parquet   --artifacts-dir artifacts   --snapshot-date 2026-03-30   --output-path scores_2026-03-30.parquet
```

## Artifacts saved

- `model.txt` — LightGBM booster
- `calibrator.joblib` — isotonic calibrator
- `feature_schema.json` — ordered feature list and fill rules
- `metrics_summary.json` — aggregate metrics
- `fold_metrics.csv` — per-fold metrics
- `backtest_monthly.csv` — monthly holdout performance
- `training_config.json` — reproducibility config

## Notes

- The scoring command uses the latest available snapshot by default.
- Feature engineering is intentionally not redefined here; this package assumes `pre.parquet` is already feature-complete.
- Missing values are handled deterministically based on fixed rules in the config.
