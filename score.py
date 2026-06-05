from __future__ import annotations

import argparse

from .pipeline import run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily batch scoring for churn model.")
    parser.add_argument("--data-path", required=True, help="Path to pre.parquet")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output-path", required=True, help="Where to write scored customers (.csv or .parquet)")
    parser.add_argument("--snapshot-date", default=None, help="Optional snapshot date, e.g. 2026-03-30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_scoring(
        data_path=args.data_path,
        artifacts_dir=args.artifacts_dir,
        output_path=args.output_path,
        snapshot_date=args.snapshot_date,
    )
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
