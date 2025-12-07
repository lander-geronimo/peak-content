"""Helper utilities shared across ETL jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import re


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure the DataFrame contains every column we expect."""

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def ensure_datetime(series: pd.Series) -> pd.Series:
    """Parse a pandas Series into timezone-aware UTC timestamps."""

    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    return timestamps


def clean_hashtag_string(raw: str | float | None) -> List[str]:
    """Split comma/space separated hashtags into lowercase tokens."""

    if not raw or pd.isna(raw):
        return []
    tags = re.split(r"[,\s]+", str(raw))
    cleaned = [tag.lower().lstrip("#") for tag in tags if tag.strip()]
    return cleaned


def create_parent_dir(path: Path) -> None:
    """Create parent directories for a file if they do not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_missing(df: pd.DataFrame, columns: Iterable[str]) -> dict[str, int]:
    """Return a dict mapping column name -> number of missing rows."""

    return {col: int(df[col].isna().sum()) for col in columns if col in df.columns}


def print_quality_report(report: dict[str, int]) -> None:
    """Pretty-print null counts so analysts see data quality issues quickly."""

    if not report:
        print("No missing-value issues detected ✅")
        return
    print("Missing values overview:")
    for col, count in sorted(report.items(), key=lambda item: item[1], reverse=True):
        print(f"- {col}: {count}")

