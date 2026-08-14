# ml-service

Python ML microservice for ResumeForge. Separate codebase and separate deploy
from `resume-optimizer` (the Next.js app) — do not assume shared files, env,
or imports between the two.

## How it fits together

- Deployed independently on Render as its own service/repo.
- Called over HTTP by the Next.js app, authenticated via an `x-api-key` header.
- Next.js app never calls into Python code directly — HTTP only.
- `GET /health` is the exception — public, no API key required. Every other
  endpoint is gated on `x-api-key`.

## Hard constraint: Render free tier memory

Render free tier caps this service at **512MB RAM**. This is a hard limit,
not a target — exceeding it gets the instance OOM-killed.

**No torch, no sentence-transformers, no neural embeddings.** They don't fit
in 512MB alongside the rest of the process. Stick to scikit-learn (TF-IDF)
and XGBoost for anything ML-related. If a dependency pulls in torch
transitively, don't add it.

## Current goal: resume↔JD match score

Training a model that scores how well a resume matches a job description.
Served via `POST /score`, called by the Next.js app's `/tailor` page.

- Approach: scikit-learn TF-IDF vectorization + XGBoost. No neural
  embeddings (see memory constraint above).
- `training/` — training scripts, not deployed with the service.
- `model/` — trained model artifacts (vectorizer, XGBoost model) loaded at
  serve time by `main.py`.

## Responsibilities (current and planned)

- Resume parsing
- Resume↔JD match scoring (`POST /score`) — in progress
- Trained ATS (Applicant Tracking System) classifier
- Resume rewrite agents
- LaTeX compilation

## Status

`/health` endpoint and API-key-gated routing pattern in place. Match-score
model (TF-IDF + XGBoost) is the active work: training pipeline in
`training/`, artifacts land in `model/`, served via `POST /score`.
