# peak-content

Designing a ML model to predict peak times and content to post on social media.

## Project Focus

This repository powers the “Best Time & Best Trend To Post” study for Instagram Reels and TikTok. The end goal is a transparent pipeline that ingests trending posts, cleans them, engineers features, and trains ML models that answer two creator-centric questions:

1. **When** should I post to maximize reach and engagement?
2. **What** topics, hashtags, or audio trends should I prioritize today?

## Commenting Guidelines

Every script in `src/` should include concise, beginner-friendly comments that explain the *why* behind each block (e.g., “convert UTC to local time so creators see recommendations in their timezone”). Avoid repeating the code literally; describe the intent and any assumptions. This keeps the project approachable for classmates who are new to data engineering or ML.

## Environment Setup

1. Create and activate a virtual environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies with `pip install -e .` (uses `pyproject.toml`).
3. Run `pre-commit install` once we add hooks (optional future step).

Add new tools/libraries directly to `pyproject.toml` so every collaborator shares the same environment.

## Data Audit Notes

Raw TikTok data currently lives in `data/raw/tiktok_merged_data_deduplicated.csv` (7,225 rows, 14 columns). A lightweight audit summary, including missing-value counts and example rows, is in `reports/data_audit.md`. Review that file before updating schemas or ETL logic so we keep assumptions aligned.

## Cleaning Raw Data

Run the ETL cleaner to standardize timestamps, hashtags, and engagement velocity, and to emit a `Parquet` artifact for modeling:

```bash
python -m src.etl.clean_tiktok --input data/raw/tiktok_merged_data_deduplicated.csv --output data/processed/posts.parquet
```

The script logs missing-value counts so we can catch upstream scraping issues quickly.

## Loading Data into PostgreSQL

1. Create a database and run the schema in `db/schema.sql`.
2. Export your connection string as `DATABASE_URL` (e.g., `postgresql://user:pass@localhost:5432/peak_content`).
3. Execute the loader: `python -m src.data.load_to_db --csv-path data/raw/tiktok_merged_data_deduplicated.csv`.

The loader inserts creators first and then posts. It uses `ON CONFLICT DO NOTHING`, so re-running it is safe and will simply skip rows that already exist.
