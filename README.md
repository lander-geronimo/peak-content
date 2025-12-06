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
