"""Clean, normalize, and enrich the raw TikTok CSV before modeling.

This script focuses on transparency: every transformation has a matching
comment so newer teammates understand the intent behind the data wrangling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from . import utils

RAW_DEFAULT = Path("data/raw/tiktok_merged_data_deduplicated.csv")
OUTPUT_DEFAULT = Path("data/processed/posts.parquet")

REQUIRED_COLUMNS = [
    "video_id",
    "author",
    "description",
    "likes",
    "comments",
    "shares",
    "plays",
    "hashtags",
    "music",
    "create_time",
    "video_url",
    "fetch_time",
]


def parse_args() -> argparse.Namespace:
    """Simple CLI to point the cleaner at different files if needed."""

    parser = argparse.ArgumentParser(description="Clean TikTok CSV export.")
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT, help="Raw CSV path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Where to store the cleaned Parquet file.",
    )
    return parser.parse_args()


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and coerce numeric types."""

    # Keep the original text but add snake_case aliases that downstream code expects.
    column_map = {
        "description": "caption",
        "create_time": "created_at",
        "fetch_time": "fetched_at",
    }
    df = df.rename(columns=column_map)

    numeric_cols = ["likes", "comments", "shares", "plays"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["author"] = df["author"].astype(str).str.strip()

    return df


def clean_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Create helpful hashtag helper columns."""

    df["hashtags_list"] = df["hashtags"].apply(utils.clean_hashtag_string)
    df["has_hashtags"] = df["hashtags_list"].apply(lambda tags: len(tags) > 0)
    df["hashtag_count"] = df["hashtags_list"].apply(len)
    return df


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure created_at/fetched_at are timezone-aware UTC timestamps."""

    df["created_at"] = utils.ensure_datetime(df["created_at"])
    df["fetched_at"] = utils.ensure_datetime(df["fetched_at"])

    # Calculate how many hours the post has been live at fetch time. We use this
    # to approximate per-hour engagement velocity.
    df["hours_live"] = (
        (df["fetched_at"] - df["created_at"]).dt.total_seconds() / 3600
    ).clip(lower=1).fillna(24)
    return df


def compute_engagement_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate engagement deltas based on the observed totals."""

    metrics = ["likes", "comments", "shares", "plays"]
    for metric in metrics:
        per_hour = df[metric] / df["hours_live"]
        df[f"{metric}_per_hour"] = per_hour
        df[f"{metric}_1h_est"] = per_hour
        df[f"{metric}_3h_est"] = per_hour * 3
        df[f"{metric}_24h_est"] = per_hour * 24
    return df


def build_quality_log(df: pd.DataFrame) -> None:
    """Print missing-value counts for the most important fields."""

    report = utils.summarize_missing(df, ["caption", "hashtags", "created_at", "plays"])
    utils.print_quality_report(report)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    utils.require_columns(df, REQUIRED_COLUMNS)
    df = df.dropna(subset=["video_id"]).drop_duplicates(subset="video_id")
    df = standardize_schema(df)
    df = clean_hashtags(df)
    df = normalize_timestamps(df)
    df = compute_engagement_deltas(df)

    build_quality_log(df)

    utils.create_parent_dir(args.output)
    df.to_parquet(args.output, index=False)
    print(f"Saved cleaned dataset to {args.output}")


if __name__ == "__main__":
    main()

