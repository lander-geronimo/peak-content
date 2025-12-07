"""Feature engineering pipeline that prepares model-ready data and trend metrics."""

from __future__ import annotations

import argparse
import json
from ast import literal_eval
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROCESSED_DEFAULT = Path("data/processed/posts.parquet")
FEATURE_OUT_DEFAULT = Path("data/features/training_set.parquet")
TREND_OUT_DEFAULT = Path("reports/trend_metrics.json")
RECENT_WINDOW_DAYS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build feature matrix and trend metrics from cleaned TikTok data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DEFAULT,
        help="Cleaned parquet/CSV input produced by src/etl/clean_tiktok.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURE_OUT_DEFAULT,
        help="Path for the feature matrix parquet file.",
    )
    parser.add_argument(
        "--trend-summary",
        type=Path,
        default=TREND_OUT_DEFAULT,
        help="Where to store the aggregated trend metrics (JSON).",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=RECENT_WINDOW_DAYS,
        help="Number of trailing days treated as 'recent' for spike calculations.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run src/etl/clean_tiktok.py first or point to a valid file."
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    return df


def ensure_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ("created_at", "fetched_at"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    if "hours_live" not in df.columns and {"created_at", "fetched_at"}.issubset(df.columns):
        df["hours_live"] = (
            (df["fetched_at"] - df["created_at"]).dt.total_seconds() / 3600
        ).clip(lower=1)
    df["hours_live"] = df["hours_live"].fillna(24)
    return df


def ensure_list_column(series: pd.Series) -> pd.Series:
    def _convert(value) -> list[str]:
        if isinstance(value, list):
            iterable = value
        elif value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        else:
            text = str(value).strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = text.strip("[]").split(",")
            else:
                parsed = text.replace("#", " ").replace(",", " ").split()
            iterable = parsed
        cleaned = [str(item).lower().lstrip("#").strip() for item in iterable if str(item).strip()]
        return cleaned

    return series.apply(_convert)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["created_hour"] = df["created_at"].dt.hour
    df["created_weekday"] = df["created_at"].dt.dayofweek
    df["is_weekend"] = df["created_weekday"].isin([5, 6]).astype(int)
    radians = 2 * np.pi * df["created_hour"] / 24
    df["posting_hour_sin"] = np.sin(radians)
    df["posting_hour_cos"] = np.cos(radians)
    bins = [-np.inf, 3, 6, 12, 24, 48, np.inf]
    labels = ["<=3h", "3-6h", "6-12h", "12-24h", "24-48h", ">48h"]
    df["hours_live_bucket"] = pd.cut(df["hours_live"], bins=bins, labels=labels).astype(str)
    return df


def add_content_features(df: pd.DataFrame) -> pd.DataFrame:
    caption = df.get("caption", "")
    caption = caption.fillna("")
    df["caption_length_chars"] = caption.str.len()
    df["caption_length_words"] = caption.str.split().apply(len)
    df["contains_question"] = caption.str.contains(r"\?", regex=True).astype(int)
    df["contains_exclamation"] = caption.str.contains("!", regex=False).astype(int)
    cta_pattern = r"(?:follow|like and share|link in bio|comment|subscribe)"
    df["contains_cta"] = caption.str.contains(cta_pattern, case=False, regex=True).astype(
        int
    )
    return df


def compute_hashtag_metrics(
    df: pd.DataFrame, recent_window_days: int
) -> tuple[pd.DataFrame, dict]:
    if "hashtags_list" not in df.columns:
        if "hashtags" in df.columns:
            df["hashtags_list"] = ensure_list_column(df["hashtags"])
        else:
            df["hashtags_list"] = [[] for _ in range(len(df))]
    else:
        df["hashtags_list"] = ensure_list_column(df["hashtags_list"])

    exploded = (
        df.loc[df["hashtags_list"].map(len) > 0]
        .explode("hashtags_list")
        .rename(columns={"hashtags_list": "hashtag"})
    )
    hashtag_counts = exploded["hashtag"].value_counts()
    df["topical_diversity"] = df["hashtags_list"].apply(lambda tags: len(set(tags)))

    def mean_count(tags: Iterable[str]) -> float:
        if not tags:
            return 0.0
        return float(np.mean([hashtag_counts.get(tag, 0) for tag in tags]))

    df["hashtag_popularity_mean"] = df["hashtags_list"].apply(mean_count)
    recent_cutoff = (df["created_at"].max() or pd.Timestamp.utcnow()) - pd.Timedelta(
        days=recent_window_days
    )
    recent_mask = df["created_at"] >= recent_cutoff
    recent_exploded = (
        df.loc[recent_mask & (df["hashtags_list"].map(len) > 0)]
        .explode("hashtags_list")
        .rename(columns={"hashtags_list": "hashtag"})
    )
    def probability(series: pd.Series) -> pd.Series:
        total = series.sum()
        if total == 0:
            return pd.Series(dtype=float)
        return series / total

    global_prob = probability(hashtag_counts)
    recent_prob = probability(recent_exploded["hashtag"].value_counts())
    trend_ratio = (recent_prob / global_prob).replace([np.inf, -np.inf], np.nan).fillna(0)
    if trend_ratio.empty:
        trend_ratio = pd.Series(dtype=float)

    def trend_score(tags: Iterable[str]) -> float:
        if not tags:
            return 0.0
        scores = [trend_ratio.get(tag, 0.0) for tag in tags]
        return float(np.mean(scores))

    df["hashtag_trend_score"] = df["hashtags_list"].apply(trend_score)
    df["has_trending_hashtag"] = (df["hashtag_trend_score"] > 1.2).astype(int)

    trend_summary = {
        "top_hashtags_overall": hashtag_counts.head(10).to_dict(),
        "top_hashtags_recent": trend_ratio.sort_values(ascending=False)
        .head(10)
        .to_dict(),
    }
    return df, trend_summary


def compute_audio_metrics(
    df: pd.DataFrame, recent_window_days: int
) -> tuple[pd.DataFrame, dict]:
    music_col = df.get("music", pd.Series("", index=df.index))
    df["music_normalized"] = music_col.fillna("unknown").str.lower().str.strip()
    audio_counts = df["music_normalized"].value_counts()
    recent_cutoff = (df["created_at"].max() or pd.Timestamp.utcnow()) - pd.Timedelta(
        days=recent_window_days
    )
    recent_audio_counts = df.loc[df["created_at"] >= recent_cutoff, "music_normalized"].value_counts()

    def ratio_lookup(name: str) -> float:
        global_count = audio_counts.get(name, 0)
        recent_count = recent_audio_counts.get(name, 0)
        if global_count == 0:
            return 0.0
        return recent_count / global_count

    df["audio_popularity"] = df["music_normalized"].map(
        lambda name: audio_counts.get(name, 0)
    )
    df["audio_trend_score"] = df["music_normalized"].map(ratio_lookup)
    df["has_trending_audio"] = (df["audio_trend_score"] > 1.2).astype(int)

    summary = {
        "top_audios_overall": audio_counts.head(10).to_dict(),
        "top_audios_recent": recent_audio_counts.head(10).to_dict(),
    }
    return df, summary


def add_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    for metric in ["likes", "comments", "shares", "plays"]:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0)
    df["engagement_total"] = df[["likes", "comments", "shares"]].sum(axis=1)
    df["engagement_rate"] = df["engagement_total"] / df["plays"].replace(0, np.nan)
    df["engagement_rate"] = df["engagement_rate"].fillna(0)
    per_hour_cols = [col for col in df.columns if col.endswith("_per_hour")]
    if per_hour_cols:
        df["velocity_mean"] = df[per_hour_cols].mean(axis=1)
    else:
        df["velocity_mean"] = df["engagement_total"] / df["hours_live"]
    df["log_plays"] = np.log1p(df["plays"])
    return df


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    candidate_metrics = [
        "plays_per_hour",
        "likes_per_hour",
        "shares_per_hour",
        "engagement_rate",
    ]
    metric_col = next((col for col in candidate_metrics if col in df.columns), None)
    if metric_col is None:
        raise ValueError(
            "Unable to locate a metric column for target creation (expected plays_per_hour, likes_per_hour, etc.)."
        )
    threshold = df[metric_col].quantile(0.75)
    df["target_metric"] = df[metric_col]
    df["viral_threshold"] = threshold
    df["is_viral"] = (df["target_metric"] >= threshold).astype(int)
    return df, float(threshold)


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "video_id",
        "author",
        "created_at",
        "created_hour",
        "created_weekday",
        "is_weekend",
        "posting_hour_sin",
        "posting_hour_cos",
        "hours_live",
        "hours_live_bucket",
        "hashtag_count",
        "topical_diversity",
        "hashtag_popularity_mean",
        "hashtag_trend_score",
        "has_trending_hashtag",
        "caption_length_chars",
        "caption_length_words",
        "contains_question",
        "contains_exclamation",
        "contains_cta",
        "audio_popularity",
        "audio_trend_score",
        "has_trending_audio",
        "likes",
        "comments",
        "shares",
        "plays",
        "engagement_total",
        "engagement_rate",
        "velocity_mean",
        "log_plays",
        "target_metric",
        "is_viral",
    ]
    existing = [col for col in preferred if col in df.columns]
    return df[existing].copy()


def save_feature_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def save_trend_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)


def build_trend_summary(
    df: pd.DataFrame,
    hashtag_summary: dict,
    audio_summary: dict,
    viral_threshold: float,
) -> dict:
    hourly_mean = df.groupby("created_hour")["target_metric"].mean().sort_values(ascending=False)
    weekday_mean = df.groupby("created_weekday")["target_metric"].mean().sort_values(ascending=False)
    summary = {
        "rows": int(len(df)),
        "features": int(len(df.columns)),
        "viral_threshold": viral_threshold,
        "date_range": {
            "min_created": str(df["created_at"].min()),
            "max_created": str(df["created_at"].max()),
        },
        "best_hours": hourly_mean.head(5).to_dict(),
        "best_weekdays": weekday_mean.head(5).to_dict(),
    }
    summary.update(hashtag_summary)
    summary.update(audio_summary)
    return summary


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    df = ensure_datetime_columns(df)
    df = add_temporal_features(df)
    df = add_content_features(df)
    df, hashtag_summary = compute_hashtag_metrics(df, args.recent_window)
    df, audio_summary = compute_audio_metrics(df, args.recent_window)
    df = add_engagement_features(df)
    df, viral_threshold = build_targets(df)
    feature_matrix = select_feature_columns(df)

    trend_summary = build_trend_summary(df, hashtag_summary, audio_summary, viral_threshold)
    save_feature_matrix(feature_matrix, args.output)
    save_trend_summary(trend_summary, args.trend_summary)
    print(
        f"Saved {len(feature_matrix)} rows with {feature_matrix.shape[1]} features to {args.output}"
    )
    print(f"Trend summary written to {args.trend_summary}")


if __name__ == "__main__":
    main()

