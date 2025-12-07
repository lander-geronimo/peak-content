"""Simple CLI dashboard summarizing best posting windows and trending topics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print best time/trend insights.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/features/training_set.parquet"),
        help="Feature matrix produced by src/features/build_features.py.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top entries to display per section.",
    )
    return parser.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run src/features/build_features.py first."
        )
    return pd.read_parquet(path)


def summarize_hours(df: pd.DataFrame, k: int) -> pd.DataFrame:
    summary = (
        df.groupby("created_hour")
        .agg(avg_target=("target_metric", "mean"), viral_rate=("is_viral", "mean"))
        .sort_values(by="avg_target", ascending=False)
        .head(k)
    )
    return summary.reset_index()


def summarize_weekdays(df: pd.DataFrame, k: int) -> pd.DataFrame:
    summary = (
        df.groupby("created_weekday")
        .agg(avg_target=("target_metric", "mean"), viral_rate=("is_viral", "mean"))
        .sort_values(by="avg_target", ascending=False)
        .head(k)
    )
    return summary.reset_index()


def summarize_hashtags(df: pd.DataFrame, k: int) -> pd.Series:
    if "hashtags_list" in df.columns:
        exploded = df.explode("hashtags_list")["hashtags_list"]
    elif "hashtags" in df.columns:
        exploded = df["hashtags"].fillna("").str.lower().str.split(",")
        exploded = exploded.explode().str.strip()
    else:
        return pd.Series(dtype=int)
    exploded = exploded.dropna()
    return exploded.value_counts().head(k)


def main() -> None:
    args = parse_args()
    df = load_features(args.features)

    print("=== Best Hours to Post ===")
    print(summarize_hours(df, args.top_k).to_string(index=False))
    print("\n=== Best Weekdays ===")
    print(summarize_weekdays(df, args.top_k).to_string(index=False))

    print("\n=== Top Hashtags in Recent Data ===")
    hashtags = summarize_hashtags(df, args.top_k)
    if hashtags.empty:
        print("No hashtag data found.")
    else:
        print(hashtags.to_string())


if __name__ == "__main__":
    main()

