"""CLI helper to score posts with the trained model. """

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Dict, Any

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODEL_DEFAULT = Path("models/random_forest.joblib")
FEATURES_DEFAULT = Path("data/features/training_set.parquet")
OUTPUT_DEFAULT = Path("reports/predictions.csv")

DROP_COLS = {"video_id", "author", "created_at", "target_metric", "is_viral"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate viral probabilities for feature rows or a single post."
    )
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT, help="Path to .joblib model.")
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_DEFAULT,
        help="Feature parquet file (same schema as training) for batch scoring.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Where to save the predictions CSV (batch mode).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to score (from the top).",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="",
        help="Comma-separated video_id(s) to score (uses --features dataset).",
    )
    parser.add_argument(
        "--print",
        dest="print_rows",
        action="store_true",
        help="Print the predictions table (useful with --video-id).",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Interactive mode: answer a few questions and get a single prediction.",
    )
    parser.add_argument(
        "--single-json",
        type=Path,
        help="Path to JSON file describing a single post (dict or list of dicts). If provided, batch scoring is skipped.",
    )
    return parser.parse_args()


def load_features(
    path: Path, limit: int | None, video_ids: Sequence[str] | None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature builder first.")
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    if video_ids:
        df = df[df["video_id"].astype(str).isin(video_ids)]
        if df.empty:
            raise ValueError(
                f"No rows found in {path} for video_id(s): {', '.join(video_ids)}"
            )
    return df


def load_single_sample(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        df = pd.DataFrame([payload])
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    meta_cols = [col for col in ["video_id", "author", "created_at"] if col in df.columns]
    meta = df[meta_cols].copy()
    y_true = df["is_viral"].copy() if "is_viral" in df.columns else None
    drop_cols = [col for col in DROP_COLS if col in df.columns]
    X = df.drop(columns=drop_cols)
    return X, meta, y_true


def prompt_for_features() -> Dict[str, Any]:
    """Collect a lightweight set of inputs from the CLI."""

    def ask_float(prompt: str, default: float) -> float:
        raw = input(f"{prompt} [{default}]: ").strip()
        return float(raw) if raw else default

    def ask_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        return int(raw) if raw else default

    def ask_bool(prompt: str, default: int) -> int:
        raw = input(f"{prompt} (0/1) [{default}]: ").strip()
        if raw == "":
            return default
        return 1 if raw in {"1", "true", "True"} else 0

    print("Provide feature values (press Enter to accept defaults if unsure).")
    likes = ask_float("Likes observed", 500.0)
    comments = ask_float("Comments observed", 30.0)
    shares = ask_float("Shares observed", 10.0)
    plays = ask_float("Plays observed", 15000.0)
    hours_live = ask_float("Hours live when captured", 3.0)

    engagement_total = likes + comments + shares
    engagement_rate = engagement_total / plays if plays else 0.0
    velocity_mean = engagement_total / hours_live if hours_live else 0.0
    log_plays = math.log(plays + 1)

    features: Dict[str, Any] = {
        "created_hour": ask_int("Posting hour (0-23)", 12),
        "created_weekday": ask_int("Weekday (0=Mon ... 6=Sun)", 3),
        "is_weekend": ask_bool("Is weekend?", 0),
        "hashtag_count": ask_int("Hashtag count", 1),
        "topical_diversity": ask_int("Topical diversity", 1),
        "has_trending_hashtag": ask_bool("Has trending hashtag?", 0),
        "has_trending_audio": ask_bool("Has trending audio?", 0),
        "caption_length_chars": ask_int("Caption length (chars)", 120),
        "caption_length_words": ask_int("Caption length (words)", 18),
        "contains_question": ask_bool("Contains question mark?", 0),
        "contains_exclamation": ask_bool("Contains exclamation?", 1),
        "contains_cta": ask_bool("Contains call-to-action?", 0),
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "plays": plays,
        "engagement_total": engagement_total,
        "engagement_rate": engagement_rate,
        "velocity_mean": velocity_mean,
        "log_plays": log_plays,
        "hours_live": hours_live,
        # Derived features with defaults
        "hashtag_popularity_mean": 40.0,
        "hashtag_trend_score": 1.0,
        "audio_popularity": 200.0,
        "audio_trend_score": 1.0,
    }
    return features


def inject_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns needed by the trained pipeline that aren't prompted directly."""
    if "created_hour" in df:
        radians = 2 * pd.Series(df["created_hour"], dtype=float) / 24.0 * (2 * np.pi)
        df["posting_hour_sin"] = np.sin(radians)
        df["posting_hour_cos"] = np.cos(radians)
    if "hours_live" in df:
        bins = [-np.inf, 3, 6, 12, 24, 48, np.inf]
        labels = ["<=3h", "3-6h", "6-12h", "12-24h", "24-48h", ">48h"]
        df["hours_live_bucket"] = pd.cut(df["hours_live"], bins=bins, labels=labels).astype(str)
    return df


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} not found. Train a model first.")

    model = joblib.load(args.model)

    if args.single_json:
        df = load_single_sample(args.single_json)
        df = inject_derived_features(df)
        X, meta, _ = prepare_features(df)
        proba = model.predict_proba(X)[:, 1]
        preds = model.predict(X)
        meta = meta.reindex(columns=["video_id", "author", "created_at"], fill_value=None)
        meta["viral_probability"] = proba
        meta["is_viral_prediction"] = preds
        print(meta.to_string(index=False))
        return

    if args.prompt:
        sample = prompt_for_features()
        df = pd.DataFrame([sample])
        df = inject_derived_features(df)
        X, _, _ = prepare_features(df)
        proba = model.predict_proba(X)[:, 1][0]
        pred = model.predict(X)[0]
        print(f"Viral probability: {proba:.3f} -> prediction={pred}")
        return

    video_ids = [vid.strip() for vid in args.video_id.split(",") if vid.strip()]
    df = load_features(args.features, args.limit, video_ids or None)
    X, meta, y_true = prepare_features(df)

    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    result = meta.copy()
    result["viral_probability"] = proba
    result["is_viral_prediction"] = preds
    if y_true is not None:
        result["is_viral_actual"] = y_true.values
        valid = ~y_true.isna()
        if valid.any():
            acc = accuracy_score(y_true[valid], preds[valid])
            f1 = f1_score(y_true[valid], preds[valid], zero_division=0)
            try:
                auc = roc_auc_score(y_true[valid], proba[valid])
            except ValueError:
                auc = float("nan")
            print(
                f"Metrics on {valid.sum()} rows with labels -> "
                f"accuracy={acc:.3f} f1={f1:.3f} roc_auc={auc:.3f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")

    if video_ids or args.print_rows:
        preview_cols = [
            col
            for col in ["video_id", "viral_probability", "is_viral_prediction"]
            if col in result.columns
        ]
        print("\nSample predictions:")
        print(result[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

