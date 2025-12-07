"""Evaluate a trained classifier against a hold-out split and create plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .train import (
    build_model_registry,
    build_preprocessor,
    load_features,
    split_features_target,
)


FEATURE_PATH_DEFAULT = Path("data/features/training_set.parquet")
METRICS_PATH_DEFAULT = Path("reports/model_metrics.json")
EVAL_REPORT_DEFAULT = Path("reports/model_eval.md")
FIGURES_DIR_DEFAULT = Path("reports/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models on hold-out data.")
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURE_PATH_DEFAULT,
        help="Path to the feature matrix parquet file.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=METRICS_PATH_DEFAULT,
        help="JSON file produced by src.models.train (used to detect best model).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override which model to evaluate (logistic_regression, random_forest, gradient_boosting).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=EVAL_REPORT_DEFAULT,
        help="Markdown file summarizing evaluation metrics.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR_DEFAULT,
        help="Directory for confusion matrix and ROC curve images.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for train/test split.",
    )
    return parser.parse_args()


def resolve_model_name(args_model_name: str | None, metrics_path: Path) -> str:
    if args_model_name:
        return args_model_name
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Pass --model-name explicitly or run src.models.train first."
        )
    with metrics_path.open() as handle:
        payload = json.load(handle)
    model_name = payload.get("best_model_name")
    if not model_name:
        raise ValueError(
            "Unable to find best_model_name in metrics JSON. Re-run training or pass --model-name."
        )
    return model_name


def evaluate_model(
    model_name: str,
    X,
    y,
    test_size: float,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    preprocessor, _, _ = build_preprocessor(X_train)
    estimator = build_model_registry(seed)[model_name]
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_scores = pipeline.decision_function(X_test)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
    }
    return metrics, y_test, y_pred, y_scores


def save_figures(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    figures_dir: Path,
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    cm_path = figures_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay(cm, display_labels=["Non-viral", "Viral"]).plot(colorbar=False)
    plt.title("Confusion Matrix (Hold-out)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    roc_path = figures_dir / "roc_curve.png"
    RocCurveDisplay.from_predictions(y_test, y_scores)
    plt.title("ROC Curve (Hold-out)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    return {"confusion_matrix": str(cm_path), "roc_curve": str(roc_path)}


def write_markdown_report(
    report_path: Path,
    metrics: dict,
    figures: dict,
    classification_text: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Evaluation",
        "",
        f"- **Model:** `{metrics['model']}`",
        f"- **Accuracy:** {metrics['accuracy']:.3f}",
        f"- **F1 Score:** {metrics['f1']:.3f}",
        f"- **ROC-AUC:** {metrics['roc_auc']:.3f}",
        f"- **Confusion Matrix:** ![]({figures['confusion_matrix']})",
        f"- **ROC Curve:** ![]({figures['roc_curve']})",
        "",
        "## Classification Report",
        "",
        "```\n" + classification_text.strip() + "\n```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_features(args.features)
    X, y = split_features_target(df)
    model_name = resolve_model_name(args.model_name, args.metrics)

    metrics, y_test, y_pred, y_scores = evaluate_model(
        model_name=model_name,
        X=X,
        y=y,
        test_size=args.test_size,
        seed=args.seed,
    )
    figures = save_figures(y_test, y_pred, y_scores, args.figures_dir)
    clf_report = classification_report(y_test, y_pred, digits=3)
    write_markdown_report(args.report, metrics, figures, clf_report)
    print(f"[evaluate] {model_name} accuracy={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} roc_auc={metrics['roc_auc']:.3f}")
    print(f"[evaluate] Report written to {args.report}")


if __name__ == "__main__":
    main()

