<!-- daca17f9-7396-4c20-af28-ecb44553f893 9522b226-3520-435a-a49c-7d5276af2cd3 -->
# End-to-End Trend & Timing Predictor Plan

## 1. Project Scaffolding

- Create `pyproject.toml` (or `requirements.txt`) plus `src/` package layout (`src/data`, `src/etl`, `src/features`, `src/models`, `src/visualization`).
- Add `notebooks/` for exploratory work and `reports/` for final insights.
- Document environment setup and workflow overview in [`README.md`](/Users/jessegraham/CompSci/peak-content/README.md).

## 2. Data Audit & Storage Schema

- Inspect `tiktok_merged_data_deduplicated.csv` to catalog available fields, missingness, and temporal coverage.
- Draft PostgreSQL schema in `db/schema.sql` covering posts, hashtags, audios, engagement snapshots; include indexes and constraints reflecting the fields discovered above.
- Provide seed/ingest script `src/data/load_to_db.py` (Pandas → SQLAlchemy) that can populate the schema from the CSV (and future scraped data).

## 3. ETL & Data Cleaning Pipeline

- Implement `src/etl/clean_tiktok.py` to standardize timestamps, drop duplicates, normalize hashtags/audio strings, and compute engagement deltas (1h, 3h, 24h) with validation logs.
- Add reusable utilities in `src/etl/utils.py` for timezone handling and quality checks; emit a `data/processed/posts.parquet` artifact.
- Outline Airflow/DAG-style automation in `src/etl/dag.md` describing the daily scrape → clean → load schedule (pseudo-code plus task dependencies).

## 4. Feature Engineering & Trend Signals

- Build `src/features/build_features.py` to derive:
- Temporal features (hour, weekday, recency buckets)
- Trend metrics (hashtag spike scores, audio usage frequency)
- Content features (caption length, keyword counts, optional sentence embeddings placeholder)
- Persist the feature matrix to `data/features/training_set.parquet` and log feature metadata.

## 5. Modeling & Evaluation

- Create `src/models/train.py` that trains Logistic Regression, Random Forest, and Gradient Boosting classifiers using scikit-learn pipelines (scaling + model).
- Include cross-validation, ROC-AUC/F1 tracking, model selection, and joblib serialization to `models/`.
- Add `src/models/evaluate.py` to generate comparison plots/metrics tables stored in `reports/model_eval.md` and `reports/figures/`.

## 6. Insights & Visualization

- Author `notebooks/eda_best_time.ipynb` to visualize engagement vs. time-of-day/day-of-week, hashtag spikes, and trend cycles, using Seaborn/Matplotlib.
- Produce `src/visualization/dashboard.py` (Streamlit or CLI) summarizing recommended posting windows and trending topics based on latest features + best model.
- Update README with usage instructions: data prep, training, and how to refresh insights for new days.

### To-dos

- [ ] Set up project structure & deps
- [ ] Inspect data & draft Postgres schema
- [ ] Implement cleaning + ETL pipeline
- [ ] Build feature engineering scripts
- [ ] Train & evaluate ML models
- [ ] Create EDA notebook & dashboard