"""Simple dashboard/CLI summary for posting window & trend insights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd

TREND_PATH_DEFAULT = Path("reports/trend_metrics.json")
FEATURE_PATH_DEFAULT = Path("data/features/training_set.parquet")


def load_trend_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src.features.build_features to regenerate trend metrics."
        )
    with path.open() as handle:
        return json.load(handle)


def load_feature_matrix(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def summarize_cli(trends: dict, features: Optional[pd.DataFrame]) -> None:
    print("=== Best Posting Windows ===")
    best_hours = trends.get("best_hours", {})
    best_days = trends.get("best_weekdays", {})
    for hour, score in list(best_hours.items())[:5]:
        print(f"- Hour {int(float(hour)):02d}: avg score {score:,.0f}")
    for day_idx, score in list(best_days.items())[:5]:
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(float(day_idx))]
        print(f"- {day_name}: avg score {score:,.0f}")

    print("\n=== Trending Hashtags (recent spikes) ===")
    for tag, ratio in list(trends.get("top_hashtags_recent", {}).items())[:10]:
        print(f"- #{tag}: spike x{ratio:.2f}")

    print("\n=== Trending Audio (recent counts) ===")
    for audio, count in list(trends.get("top_audios_recent", {}).items())[:10]:
        print(f"- {audio}: {count} recent uses")

    if features is not None:
        viral_by_hour = (
            features.groupby("created_hour")["is_viral"].mean().sort_values(ascending=False)
        )
        print("\n=== Viral probability by hour (from feature matrix) ===")
        for hour, prob in viral_by_hour.head(5).items():
            print(f"- Hour {int(hour):02d}: P(viral)={prob:.2%}")


def render_streamlit(trends: dict, features: Optional[pd.DataFrame]) -> None:
    import streamlit as st

    st.set_page_config(page_title="Peak Content Dashboard", layout="wide")
    st.title("Best Time & Trend Dashboard")
    st.caption(
        "Powered by cleaned TikTok data • run `src.features.build_features` to refresh snapshots."
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric("Rows analyzed", f"{trends.get('rows', 0):,}")
    metric_cols[1].metric("Feature count", f"{trends.get('features', 0):,}")
    metric_cols[2].metric("Viral threshold", f"{trends.get('viral_threshold', 0):,.0f}")

    st.subheader("Top Posting Hours")
    best_hours = pd.DataFrame(trends.get("best_hours", {}).items(), columns=["hour", "score"])
    best_hours["hour"] = best_hours["hour"].astype(float).astype(int)
    best_hours = best_hours.sort_values("score", ascending=False)
    st.bar_chart(best_hours.set_index("hour"))

    st.subheader("Top Posting Weekdays")
    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    best_days = (
        pd.DataFrame(trends.get("best_weekdays", {}).items(), columns=["weekday", "score"])
        .assign(weekday=lambda df: df["weekday"].astype(float).astype(int))
        .assign(label=lambda df: df["weekday"].map(weekday_map))
        .sort_values("score", ascending=False)
    )
    st.bar_chart(best_days.set_index("label"))

    st.subheader("Trending hashtags")
    tags_df = pd.DataFrame(
        trends.get("top_hashtags_recent", {}).items(), columns=["hashtag", "trend_ratio"]
    ).sort_values("trend_ratio", ascending=False)
    st.dataframe(tags_df.head(15))

    st.subheader("Trending audio")
    audio_df = pd.DataFrame(
        trends.get("top_audios_recent", {}).items(), columns=["audio", "recent_uses"]
    ).sort_values("recent_uses", ascending=False)
    st.dataframe(audio_df.head(15))

    if features is not None:
        st.subheader("Viral probability by hour")
        viral_hour = (
            features.groupby("created_hour")["is_viral"]
            .mean()
            .reset_index()
            .rename(columns={"is_viral": "viral_probability"})
        )
        st.line_chart(viral_hour.set_index("created_hour"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize best posting times and trending topics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trend-path", type=Path, default=TREND_PATH_DEFAULT)
    parser.add_argument("--feature-path", type=Path, default=FEATURE_PATH_DEFAULT)
    parser.add_argument(
        "--mode",
        choices=["auto", "cli", "streamlit"],
        default="auto",
        help="auto -> streamlit if running via `streamlit run`, otherwise cli summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trends = load_trend_metrics(args.trend_path)
    features = load_feature_matrix(args.feature_path)

    streamlit_env = os.environ.get("STREAMLIT_SERVER_RUNNING") == "1"
    mode = args.mode
    if mode == "auto":
        mode = "streamlit" if streamlit_env else "cli"

    if mode == "streamlit":
        render_streamlit(trends, features)
    else:
        summarize_cli(trends, features)


if __name__ == "__main__":
    main()

