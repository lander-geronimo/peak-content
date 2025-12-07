"""Utility script to move the raw TikTok CSV into the Postgres schema.

The goal is to keep the logic extremely readable so teammates who are new to
data engineering can follow along. Each step (loading the CSV, inserting
creators, then posts) is broken into small helper functions with comments that
describe *why* we do something, not just *what* the code does.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


DEFAULT_CSV_PATH = Path("data/raw/tiktok_merged_data_deduplicated.csv")


@dataclass
class LoaderConfig:
    """Simple container for CLI/env configuration."""

    csv_path: Path
    database_url: str


def parse_args() -> LoaderConfig:
    """Read CLI flags and fall back to environment variables when possible."""

    parser = argparse.ArgumentParser(
        description=(
            "Load TikTok post metadata from the CSV file into the Postgres schema."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Location of the deduplicated TikTok CSV file.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv("DATABASE_URL"),
        help="SQLAlchemy-style database URL, e.g. postgres://user:pass@host/db.",
    )
    args = parser.parse_args()

    if not args.database_url:
        raise SystemExit(
            "DATABASE_URL is not provided. Set the env var or pass --database-url."
        )

    return LoaderConfig(csv_path=args.csv_path, database_url=args.database_url)


def build_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine that manages database connections for us."""

    return create_engine(database_url, pool_pre_ping=True)


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load the CSV with pandas and normalize a few obvious columns."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    df = pd.read_csv(csv_path)

    # Trim whitespace and make sure there are no duplicate video ids.
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset="video_id").reset_index(drop=True)

    # Normalize timestamp columns; assume UTC when timezone info is missing.
    for col in ("create_time", "fetch_time", "posted_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # Replace empty strings with actual NaN values so pandas treats them as missing.
    df = df.replace({"": pd.NA})

    return df


def upsert_creators(df: pd.DataFrame, engine: Engine) -> Dict[str, int]:
    """Insert creators (authors) and return a mapping handle -> creator_id.

    TikTok rows store the author handle in the `author` column. We write each
    unique handle into the `creators` table. ON CONFLICT makes this idempotent,
    so running the loader multiple times will not duplicate entries.
    """

    creators = (
        df.loc[df["author"].notna(), ["author"]]
        .drop_duplicates()
        .rename(columns={"author": "handle"})
    )
    creators["platform"] = "tiktok"

    inserted = 0
    with engine.begin() as conn:
        for record in creators.to_dict(orient="records"):
            result = conn.execute(
                text(
                    """
                    INSERT INTO creators (handle, platform)
                    VALUES (:handle, :platform)
                    ON CONFLICT (handle) DO NOTHING
                    """
                ),
                record,
            )
            inserted += result.rowcount or 0

        # Pull the ids back so we can map posts -> creators.
        rows = conn.execute(
            text("SELECT creator_id, handle FROM creators WHERE platform = 'tiktok'")
        ).mappings()
        lookup = {row["handle"]: row["creator_id"] for row in rows}

    print(f"[creators] inserted {inserted} new rows (total handles: {len(lookup)})")
    return lookup


def format_post_rows(df: pd.DataFrame, creator_lookup: Dict[str, int]) -> List[dict]:
    """Transform the pandas DataFrame into dictionaries ready for SQL inserts."""

    records: List[dict] = []
    for _, row in df.iterrows():
        handle = row.get("author")
        creator_id = creator_lookup.get(handle)
        if creator_id is None:
            # Skip rows we cannot map to a creator (should be rare).
            continue

        records.append(
            {
                "video_id": row["video_id"],
                "creator_id": creator_id,
                "caption": row.get("description"),
                "raw_hashtags": row.get("hashtags"),
                "video_url": row.get("video_url"),
                "source_platform": "tiktok",
                "created_at": row.get("create_time"),
                "fetched_at": row.get("fetch_time"),
            }
        )
    return records


def insert_posts(records: Iterable[dict], engine: Engine) -> None:
    """Insert post records while ignoring duplicates via ON CONFLICT."""

    if not records:
        print("[posts] no records to insert")
        return

    with engine.begin() as conn:
        total = 0
        for chunk_start in range(0, len(records), 500):
            chunk = records[chunk_start : chunk_start + 500]
            conn.execute(
                text(
                    """
                    INSERT INTO posts (
                        video_id,
                        creator_id,
                        caption,
                        raw_hashtags,
                        video_url,
                        source_platform,
                        created_at,
                        fetched_at
                    )
                    VALUES (
                        :video_id,
                        :creator_id,
                        :caption,
                        :raw_hashtags,
                        :video_url,
                        :source_platform,
                        :created_at,
                        :fetched_at
                    )
                    ON CONFLICT (video_id) DO NOTHING
                    """
                ),
                chunk,
            )
            total += len(chunk)
    print(f"[posts] attempted to insert {total} rows (duplicates skipped automatically)")


def main() -> None:
    """End-to-end loader entry point."""

    config = parse_args()
    engine = build_engine(config.database_url)
    df = load_raw_csv(config.csv_path)
    creators = upsert_creators(df, engine)
    post_records = format_post_rows(df, creators)
    insert_posts(post_records, engine)
    print("Done! Posts are now stored in the database.")


if __name__ == "__main__":
    main()

