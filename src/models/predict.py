"""CLI helper to score posts with the trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_DEFAULT = Path("models/random_forest.joblib")
FEATURES_DEFAULT = Path("data/features/training_set.parquet")
OUTPUT_DEFAULT = Path("reports/predictions.csv")

DROP_COLS = {"video_id", "author", "created_at", "target_metric", "is_viral"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate viral probabilities for feature rows using the trained model.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT, help="Path to .joblib model.")
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_DEFAULT,
        help="Feature parquet file (same schema as training).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Where to save the predictions CSV (includes metadata columns).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to score (from the top).",
    )
    return parser.parse_args()


def load_features(path: Path, limit: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run feature builder first.")
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    return df


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    meta_cols = [col for col in ["video_id", "author", "created_at"] if col in df.columns]
    meta = df[meta_cols].copy()
    y_true = df["is_viral"].copy() if "is_viral" in df.columns else None
    X = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
    return X, meta, y_true


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"{args.model} not found. Train a model first.")

    df = load_features(args.features, args.limit)
    X, meta, y_true = prepare_features(df)

    model = joblib.load(args.model)
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    result = meta.copy()
    result["viral_probability"] = proba
    result["is_viral_prediction"] = preds
    if y_true is not None:
        result["is_viral_actual"] = y_true.values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main() 

