"""Train baseline classifiers to predict whether a TikTok post will go viral."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


FEATURE_PATH_DEFAULT = Path("data/features/training_set.parquet")
MODEL_DIR_DEFAULT = Path("models")
REPORT_PATH_DEFAULT = Path("reports/model_metrics.json")
TARGET_COLUMN = "is_viral"


@dataclass
class TrainConfig:
    feature_path: Path
    model_dir: Path
    report_path: Path
    cv_folds: int
    random_state: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train classifiers on feature matrix.")
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURE_PATH_DEFAULT,
        help="Path to the training feature parquet file.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR_DEFAULT,
        help="Directory where trained pipelines will be stored.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH_DEFAULT,
        help="Location for the JSON metrics summary.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits for cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    return TrainConfig(
        feature_path=args.features,
        model_dir=args.model_dir,
        report_path=args.report,
        cv_folds=args.cv_folds,
        random_state=args.seed,
    )


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/features/build_features.py first.")
    return pd.read_parquet(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} column missing from feature matrix.")
    drop_cols = {"video_id", "author", "created_at", "target_metric"}
    existing_drop = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=existing_drop)
    y = X.pop(TARGET_COLUMN)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))
    if not transformers:
        raise ValueError("Feature matrix does not contain numeric or categorical columns.")
    preprocessor = ColumnTransformer(transformers)
    return preprocessor, numeric_cols, categorical_cols


def build_model_registry(random_state: int) -> Dict[str, object]:
    """Return the three baseline estimators we want to compare."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
            random_state=random_state,
        ),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    config: TrainConfig,
) -> list[dict]:
    registry = build_model_registry(config.random_state)
    cv = StratifiedKFold(
        n_splits=config.cv_folds, shuffle=True, random_state=config.random_state
    )
    scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
    metrics: list[dict] = []

    for name, estimator in registry.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        cv_scores = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=None, return_train_score=False
        )
        summary = {
            "model": name,
            "roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
            "f1": float(np.mean(cv_scores["test_f1"])),
            "accuracy": float(np.mean(cv_scores["test_accuracy"])),
        }
        metrics.append(summary)
        print(f"[CV] {name}: roc_auc={summary['roc_auc']:.3f}, f1={summary['f1']:.3f}, acc={summary['accuracy']:.3f}")
    return metrics


def train_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    metrics: list[dict],
    config: TrainConfig,
) -> tuple[Path, str]:
    best = max(metrics, key=lambda item: item["roc_auc"])
    model_name = best["model"]
    estimator = build_model_registry(config.random_state)[model_name]
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X, y)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[train] Saved best model '{model_name}' to {model_path}")
    return model_path, model_name


def write_report(
    metrics: list[dict], report_path: Path, best_model_path: Path, best_model_name: str
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_model_path": str(best_model_path),
        "best_model_name": best_model_name,
        "metrics": metrics,
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[report] Metrics written to {report_path}")


def main() -> None:
    config = parse_args()
    df = load_features(config.feature_path)
    X, y = split_features_target(df)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    print(f"[data] Loaded {len(df)} rows with {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features.")
    metrics = evaluate_models(X, y, preprocessor, config)
    best_model_path, best_model_name = train_best_model(X, y, preprocessor, metrics, config)
    write_report(metrics, config.report_path, best_model_path, best_model_name)


if __name__ == "__main__":
    main()

