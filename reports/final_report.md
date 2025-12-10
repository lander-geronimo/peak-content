# Final Report: Best Time & Best Trend To Post (TikTok/Instagram)

*Date:* December 9, 2025  
*Team:* [Your Name], [Teammate Name]  
*Repository:* `/Users/jessegraham/CompSci/peak-content`

---

## 1. Project Definition (Problem & Strategy)

**Problem.** We aim to recommend (a) the best time to post and (b) the most relevant trending signals (hashtags, audio) for short‑form content on TikTok/Instagram. Creators struggle with opaque platform analytics and generic “best time” advice. We want a transparent, reproducible pipeline that ingests public post data, cleans it, engineers trend‑aware features, trains an ML model to estimate virality, and surfaces insights through a dashboard and a prediction CLI.

**Strategic aspects.**
- Creator‑centric: actionable hours and trending tags/audio they can use today.
- Transparent & reproducible: open code, explicit feature engineering, clear evaluation.
- Daily refreshable: designed so ingest/clean/feature/model steps can be rerun frequently.

**Course alignment.**  
Relational schema design (db/schema.sql), ETL and data cleaning (pandas, timezone normalization), feature scaling/encoding, model evaluation (CV, ROC‑AUC/F1), overfitting mitigation, and visualization (matplotlib/seaborn, Streamlit) all mirror lecture topics on data pipelines, supervised learning, validation, and communication.

---

## 2. Novelty & Importance

**Why it matters.** Short‑form platforms are crowded, and creators need data‑backed timing and trend guidance rather than generic blog advice. Most existing tools are proprietary and opaque, so we focused on a consumer‑friendly, explainable pipeline from raw public data to recommendations that can be rerun daily. In current practice, “best time” dashboards hide their data and assumptions, insights are locked to specific platforms, and hashtag/audio trend detection is rarely transparent. Our approach keeps everything open and reproducible—feature parquet, trend JSON, predictions CSV, and code—and computes trend spikes using recent‑versus‑global ratios to surface emerging signals. The pipeline is cohesive end‑to‑end (ingest → clean → feature → train/eval → predict → dashboard) and includes a documented overfitting pivot from Random Forest to Logistic Regression. Related industry and academic work touches on time‑aware recommendation, but few open, creator‑oriented pipelines with daily trend signals exist; we aim to fill that gap with something practical and easy to rerun.

---

## 3. Data, ETL, and Feature Engineering

### 3.1 Data source
Raw CSV: `data/raw/tiktok_merged_data_deduplicated.csv` (7,225 rows, 14 columns) with `video_id`, `author`, `description`, `hashtags`, `music`, engagement counts (`likes/comments/shares/plays`), `create_time`, `fetch_time`, `video_url`. No private data.
Data audit observations: hashtags are sparse and noisy; timestamps vary in format/timezone; some engagement fields are zero/NaN; a handful of duplicate `video_id`s existed and were dropped; outliers in plays/likes were retained to preserve true viral events.

### 3.2 ETL (src/etl/clean_tiktok.py)
The ETL step enforces a consistent schema, trustworthy timestamps, per-hour engagement rates, and early visibility into data quality issues. We rename `create_time` to `created_at` and `fetch_time` to `fetched_at`, coerce numerics with `errors="coerce"`, trim `video_id/author`, drop rows missing `video_id`, and de‑duplicate on `video_id`. Hashtags are cleaned by splitting on commas/spaces, stripping `#`, and lowercasing; empty or NaN values become empty lists, while the raw text is kept for debugging. Timestamps are parsed to UTC, `hours_live` is computed as `(fetched_at − created_at)` in hours and clipped to at least 1 to avoid division by zero, and missing values are set to 24 to avoid inflating per-hour rates. We derive per‑hour likes/comments/shares/plays along with 1h/3h/24h projections to support velocity features and sanity checks. Finally, we log missing counts for captions, hashtags, created_at, and plays to flag scraping gaps before training.

Key snippet:
```python
# src/etl/clean_tiktok.py
df["hashtags_list"] = df["hashtags"].apply(utils.clean_hashtag_string)
df["created_at"] = utils.ensure_datetime(df["created_at"])
df["fetched_at"] = utils.ensure_datetime(df["fetched_at"])
df["hours_live"] = ((df["fetched_at"] - df["created_at"]).dt.total_seconds() / 3600).clip(lower=1)
for metric in ["likes", "comments", "shares", "plays"]:
    per_hour = df[metric] / df["hours_live"]
    df[f"{metric}_per_hour"] = per_hour
```

Run:  
`python -m src.etl.clean_tiktok --input data/raw/tiktok_merged_data_deduplicated.csv --output data/processed/posts.parquet`

### 3.3 Feature Engineering (src/features/build_features.py)
Our feature set is designed to capture temporal patterns, content cues, and real‑time trend spikes, while keeping the label balanced and the features interpretable enough to refresh daily. Temporal features include `created_hour`, `created_weekday`, a weekend flag, sine/cosine transforms of hour to respect the cyclic nature of time, and an `hours_live_bucket` to coarse‑grain exposure time; this reflects audience availability and feed recency effects without treating hour 0 and 23 as far apart. Content features measure caption length in characters and words, flag questions or exclamations, and detect CTAs via regex, acknowledging that concise, directive captions can prompt engagement and punctuation signals intent. Trend features are split between hashtags and audio: hashtags are exploded, globally counted, and compared in a recent window via probability ratios to detect spikes; we compute `hashtag_trend_score` (mean per‑tag spike), `has_trending_hashtag`, topical diversity, and mean popularity to capture both momentum and breadth. Audio is normalized and counted with recent/global ratios to produce `audio_trend_score` and `has_trending_audio`, reflecting how trending audio drives discovery on TikTok. Engagement features include totals, `engagement_rate` (interactions/plays), `velocity_mean` from per‑hour metrics, and `log_plays` to stabilize heavy tails—rates and velocity are more informative than raw counts for early performance. The label `is_viral` is defined as the top 25% of `plays_per_hour`, a shift from an earlier, too‑sparse top‑10% threshold to improve class balance and stability while still focusing on high performers.

Outputs:  
- `data/features/training_set.parquet` (33 features + label)  
- `reports/trend_metrics.json` (best hours/days, top hashtags/audio, viral threshold)

Key snippet:
```python
# src/features/build_features.py
df["created_hour"] = df["created_at"].dt.hour
df["caption_length_words"] = df["caption"].fillna("").str.split().apply(len)
df["has_trending_hashtag"] = (df["hashtag_trend_score"] > 1.2).astype(int)
df["audio_trend_score"] = df["music_normalized"].map(ratio_lookup)
threshold = df["plays_per_hour"].quantile(0.75)
df["is_viral"] = (df["plays_per_hour"] >= threshold).astype(int)
```

Run:  
`python -m src.features.build_features --input data/processed/posts.parquet --output data/features/training_set.parquet --trend-summary reports/trend_metrics.json`

### 3.4 EDA (notebooks/eda_best_time.ipynb)
- Hour‑of‑day performance (log scale) → peaks around 9–11 UTC.
- Weekday performance with dual y‑axis (mean target vs. P(is_viral)) → Monday/Tuesday strongest.
- Trending hashtags/audio tables (using processed data + merged labels).
- Path‑aware so it runs from repo root or /notebooks.

---

## 4. Modeling, Evaluation, and Prediction

### 4.1 Models tested (src/models/train.py)
- Logistic Regression (balanced, max_iter=500)
- Random Forest (300 trees, class_weight balanced_subsample)
- Gradient Boosting (300 estimators)

Key snippet:
```python
# src/models/train.py
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
])
models = {
    "logistic_regression": LogisticRegression(max_iter=500, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample"),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05),
}
```

**Validation setup:** We used stratified 5-fold cross‑validation on the training set with a shared preprocessor (impute/scale for numeric columns and impute/one‑hot for categoricals) to ensure fairness across models. Metrics (ROC‑AUC, F1, accuracy) were averaged over folds, and hyperparameters were intentionally modest to limit variance.

### 4.2 Overfitting pivot
Random Forest and Gradient Boosting returned 1.0 scores in both CV and hold‑out. Given the dataset’s size and homogeneity, we interpreted this as overfitting rather than genuine superiority. To improve generalization, we designated Logistic Regression as the final model and kept RF/GB only as documented overfit baselines.

### 4.3 Evaluation (src/models/evaluate.py)
After cross‑validation, we performed a stratified train/test split and reported Accuracy, F1, and ROC‑AUC, along with a confusion matrix and ROC curve (`reports/figures`, summarized in `reports/model_eval.md`). The perfect scores likely reflect dataset homogeneity; in a production setting we would enforce temporal splits or test on newer scrapes to stress‑test generalization.

### 4.4 Prediction CLI (src/models/predict.py)
- Loads saved pipeline (preprocessor + classifier), scores any feature parquet.
- Writes `reports/predictions.csv` with `viral_probability`, `is_viral_prediction`, and, if labels present, `is_viral_actual`.
- Prints batch metrics (accuracy/F1/ROC‑AUC) when `is_viral` exists in the input.

Run example:  
`python -m src.models.predict --model models/random_forest.joblib --features data/features/training_set.parquet --output reports/predictions.csv`

### 4.5 Dashboard (src/visualization/dashboard.py)
- CLI mode: `python -m src.visualization.dashboard --mode cli`
- Streamlit UI: `/Users/jessegraham/Library/Python/3.9/bin/streamlit run src/visualization/dashboard.py -- --mode streamlit`
- Shows best hours/days, trending hashtags/audio, viral probability by hour (from `trend_metrics.json` and the feature matrix).

---

## 5. Experiments: Failures, Pivots, and Successes

We deliberately iterated through multiple approaches and documented the failures that led to a more stable pipeline. The first hashtag parser split only on commas, collapsing space‑delimited tags; by switching to a regex that handled both commas and spaces, lowercasing, and stripping `#`, we recovered diversity and popularity signals that were previously hidden. Early trend detection relied on global counts and missed emerging spikes; introducing recent‑window probability ratios and explicit trend flags let us surface sudden hashtag/audio spikes instead of static “top” tags. Our first label definition used the top 10\% of plays/hour, creating a tiny positive class and unstable models; redefining `is_viral` as the top 25\% balanced the label while keeping a “high performer” focus. Random Forest and Gradient Boosting both scored 1.0 on CV and hold‑out, which on this modest, homogeneous dataset signaled overfitting; we pivoted to Logistic Regression as the final model and kept RF/GB only as overfit baselines. Practical issues also arose: running the notebook from /notebooks broke paths, so we added root resolution; the feature matrix lacked hashtags, leaving viral-only plots empty, so we merged `processed` hashtags with `is_viral` by `video_id`; matplotlib/fontconfig failed to build caches in the sandbox, so we used writable `.matplotlib`/`.cache` dirs and documented the env vars; and raw per-hour heuristics were noisy, prompting log-scale visuals and reliance on engineered temporal features plus model aggregation rather than naive averages.

---

## 6. Results & Findings

### 6.1 Best time signals (dataset-specific)
The log-scaled hour-of-day analysis shows 9–11 UTC as the strongest window, likely aligning with morning or early workday scrolling for a broad audience. By day of week, Monday and Tuesday carry slightly higher mean engagement and viral probability; weekends were weaker than expected in this snapshot, suggesting audience routines matter and should be rechecked with fresher data.

### 6.2 Trend signals
Hashtags with long, event-like strings spike in the recent window (often more than 3× global), validating the recent/global ratio approach to capture emergent topics rather than static popularity. Audio usage is dominated by “original sound” families, with a few named tracks showing recent spikes; this underscores that audio is a strong discovery surface even when hashtags are sparse.

### 6.3 Model performance (caveat: small, homogeneous dataset)
Logistic Regression, chosen as the final model, delivers high ROC‑AUC and F1 without the blatant overfit observed in RF/GB; its coefficients remain interpretable for future feature-importance work. RF/GB achieved perfect metrics but were treated as overfit and retained only to illustrate why simpler, regularized models can be preferable on modest, homogeneous datasets.

### 6.4 Prediction outputs
The predictions file (`reports/predictions.csv`) carries per-post probabilities and labels; when the input includes ground truth (`is_viral`), the CLI also prints batch metrics (e.g., “Metrics on 7225 rows … accuracy=1.000 f1=1.000 roc_auc=1.000”) as a sanity check. For unseen data, only probabilities and predicted labels are emitted.

---

## 7. Advantages and Limitations

**Advantages**  
The pipeline is transparent and reproducible end to end: code, feature parquets, trend summaries, and predictions are open and can be rerun. The engineered features align with what creators care about—time windows, trending hashtags/audio, caption cues, and engagement signals—making recommendations intuitive. The design is refreshable on a daily cadence by rerunning ETL, feature building, and modeling. On top of that, we provide a simple prediction CLI for batch scoring and a dual-mode dashboard (CLI and Streamlit) so users can consume insights in whichever interface they prefer.

**Limitations**  
We currently operate on a single-platform, single-snapshot dataset, so metrics may be inflated by the limited size and homogeneity. There is no personalization by creator/account; labels based on plays per hour can be noisy proxies for true reach. Overfitting risk persists without temporal validation—Random Forest and Gradient Boosting overfit badly in our tests—and we lack external ground truth for “reach” versus “plays,” as well as any cross-platform generalization.

---

## 8. Changes After Proposal & Oral Exam

We pivoted the final model to Logistic Regression after the oral exam revealed that the perfect scores from Random Forest were likely overfitting. We fixed trend/hashtag handling by properly exploding and merging so viral-only plots would display, and we updated the notebook and scripts to resolve paths from the project root, avoiding FileNotFound errors. We also documented writable cache directories for matplotlib/fontconfig to avoid runtime issues. Throughout these changes, we kept the core goal—best time plus trend insights plus prediction and dashboard—while shifting emphasis toward generalization and reproducibility.

---

## 9. How to Run (for graders)

1) **Clean:**  
`python -m src.etl.clean_tiktok --input data/raw/tiktok_merged_data_deduplicated.csv --output data/processed/posts.parquet`

2) **Feature build:**  
`python -m src.features.build_features --input data/processed/posts.parquet --output data/features/training_set.parquet --trend-summary reports/trend_metrics.json`

3) **Train (includes LR/RF/GB; final choice = LR due to overfit concerns):**  
`python -m src.models.train --features data/features/training_set.parquet --model-dir models --report reports/model_metrics.json`

4) **Evaluate:**  
`MPLCONFIGDIR=.matplotlib python -m src.models.evaluate --features data/features/training_set.parquet --metrics reports/model_metrics.json --report reports/model_eval.md --figures-dir reports/figures`

5) **Predict:**  
`python -m src.models.predict --model models/random_forest.joblib --features data/features/training_set.parquet --output reports/predictions.csv`  
Switch `--features` to any new feature parquet; labels optional. Metrics print if `is_viral` present.

6) **Dashboard:**  
CLI: `python -m src.visualization.dashboard --mode cli`  
Streamlit: `/Users/jessegraham/Library/Python/3.9/bin/streamlit run src/visualization/dashboard.py -- --mode streamlit`

7) **EDA:** Open `notebooks/eda_best_time.ipynb` (handles cwd from repo root or /notebooks).

---

## 10. Future Work
- Collect larger, time-separated datasets; add strict temporal validation. We plan to scrape additional weeks or months of TikTok data to capture seasonality and reduce overfitting. With time-stamped folds, we can evaluate how well the model generalizes to genuinely unseen days rather than random splits, while monitoring trend drift and recalibrating thresholds as the platform evolves.
- Personalize by creator/account segments; incorporate follower/timezone signals. Segmenting posts by creator characteristics (follower count, niche) and applying timezone-aware posting windows would make recommendations user-specific. Adding audience signals (e.g., inferred location or follower distribution) could benchmark strategies for small vs. large accounts and increase relevance.
- Add SHAP/feature importance and calibrated probabilities. Post-hoc explanations (e.g., SHAP values) would show which factors (time, hashtags, audio) drive viral probability, improving trust and actionability. Calibrating predicted probabilities would align scores with real-world odds, leading to better decisions.
- Deploy Streamlit publicly; schedule daily refresh; containerize for reproducibility. Publishing the dashboard (e.g., Streamlit Cloud) would remove local setup friction. A scheduled job (cron/Airflow) could rerun ETL → features → model → dashboard daily to keep trends current, and containerization (Docker) would ensure consistent environments for local and hosted runs.
- Explore sequence/time-series or survival models for trend drift and post longevity. Moving beyond static classifiers, sequence models could track evolving hashtags/audio, and survival analysis could estimate post “lifetimes.” This would capture trend decay, improve timing guidance, and quantify how long a trend remains useful.

---

## 11. Conclusion

We delivered an end-to-end, transparent pipeline that ingests public TikTok data, cleans and engineers trend-aware features, trains and evaluates multiple classifiers, pivots away from an overfitting model to a simpler Logistic Regression, and exposes insights through both a dashboard and a prediction CLI. We documented failures and pivots (hashtag parsing, trend spikes, target definition, RF overfit) and produced reproducible artifacts (feature parquet, trend JSON, metrics reports, predictions CSV). With more data and temporal validation, the system can better generalize and support daily, creator-friendly recommendations on when and what to post.

---

## 12. PDF Export (for submission)
To generate a PDF with the embedded code snippets and images:
```bash
cd /Users/jessegraham/CompSci/peak-content
pandoc reports/final_report.md -o reports/final_report.pdf --from markdown --to pdf --highlight-style tango
```
Ensure `reports/figures/confusion_matrix.png` and `reports/figures/roc_curve.png` are present; add any additional screenshots (dashboard, code) under `reports/figures/` and reference them in this markdown before exporting.

