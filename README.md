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

## Feature Engineering

Transform cleaned posts into a model-ready feature matrix and daily trend summary:

```bash
python -m src.features.build_features \
  --input data/processed/posts.parquet \
  --output data/features/training_set.parquet \
  --trend-summary reports/trend_metrics.json
```

The output file contains 33 engineered columns plus the binary label `is_viral` (top quartile of plays-per-hour). Re-run this command whenever new cleaned data is available.

## Loading Data into PostgreSQL

1. Create a database and run the schema in `db/schema.sql`.
2. Export your connection string as `DATABASE_URL` (e.g., `postgresql://user:pass@localhost:5432/peak_content`).
3. Execute the loader: `python -m src.data.load_to_db --csv-path data/raw/tiktok_merged_data_deduplicated.csv`.

The loader inserts creators first and then posts. It uses `ON CONFLICT DO NOTHING`, so re-running it is safe and will simply skip rows that already exist.

## Model Training & Evaluation

Train Logistic Regression, Random Forest, and Gradient Boosting classifiers, log cross-validation metrics, and persist the best pipeline (default: `models/random_forest.joblib`):

```bash
python -m src.models.train \
  --features data/features/training_set.parquet \
  --model-dir models \
  --report reports/model_metrics.json
```

Generate a hold-out evaluation summary (figures omitted by default to keep the repo light):

```bash
python -m src.models.evaluate \
  --features data/features/training_set.parquet \
  --metrics reports/model_metrics.json \
  --report reports/model_eval.md \
  --figures-dir reports/figures
```

Add `--generate-figures` (and optionally `MPLCONFIGDIR=.matplotlib` on macOS sandboxes) to save PNGs for presentations. By default the script prints where to regenerate them without committing large binaries.

## Exploratory Analysis & Insights

- Explore engagement vs. time/day and trending hashtags/audio in `notebooks/eda_best_time.ipynb`.
- Print a CLI “dashboard” summary of best posting windows and top hashtags/audio:

```bash
python -m src.visualization.dashboard --features data/features/training_set.parquet
```

Adjust `--top-k` to show more rows. This command simply reads the feature matrix and reports the highest-performing hours, weekdays, and hashtag counts.
