# ml-service

Python ML microservice for ResumeForge. Separate codebase and separate deploy
from `resume-optimizer` (the Next.js app) — do not assume shared files, env,
or imports between the two.

## How it fits together

- Deployed independently on Render as its own service/repo.
- Called over HTTP by the Next.js app, authenticated via an `x-api-key` header.
- Next.js app never calls into Python code directly — HTTP only.

## Responsibilities (current and planned)

- Resume parsing
- Embeddings generation
- Trained ATS (Applicant Tracking System) classifier
- Resume rewrite agents
- LaTeX compilation

## Status

Plumbing only right now: a `/health` endpoint and API-key-gated routing pattern.
No database, no ML models wired up yet.
