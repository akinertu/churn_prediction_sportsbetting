from __future__ import annotations

import argparse
import json

from .config import TrainingConfig
from .pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model with time-aware validation.")
    parser.add_argument("--data-path", required=True, help="Path to pre.parquet")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--train-window-days", type=int, default=365)
    parser.add_argument("--n-folds", type=int, default=4)
    parser.add_argument("--val-window-days", type=int, default=30)
    parser.add_argument("--gap-days", type=int, default=3)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="isotonic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        train_window_days=args.train_window_days,
        n_folds=args.n_folds,
        val_window_days=args.val_window_days,
        gap_days=args.gap_days,
        calibration_method=args.calibration_method,
    )
    result = run_training(config)
    print(json.dumps(result["metrics_summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
