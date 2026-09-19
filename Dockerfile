# Python 3.11 to match the training CI (.github/workflows/train.yml), which is
# where model/match_model.joblib is pickled. -slim is Debian-based: the
# scikit-learn / scipy / xgboost-cpu manylinux wheels install as-is, with no
# compiler needed. (Alpine/musl would have no matching wheels and would force a
# slow source build.)
FROM python:3.11-slim

# Don't write .pyc files into the image (they're never reused across runs), and
# don't buffer stdout/stderr so uvicorn logs show up live in `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, in their own layer: requirements.txt changes far less often
# than main.py/scoring.py, so Docker reuses this cached layer (and skips the
# ~200MB install) on every code-only rebuild.
#
# requirements.txt ONLY -- never training/requirements.txt, which pulls
# datasets/pandas/pyarrow and, on darwin, the CUDA-bundling `xgboost` package.
# Installing this file verbatim is what preserves the xgboost-cpu pin.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Serving code only. training/ is not deployed (see CLAUDE.md); venv/, .git/ and
# __pycache__/ are excluded by .dockerignore.
COPY main.py scoring.py ./

# The trained artifact (1.3MB). scoring.py resolves it as
# Path(__file__).parent / "model" / "match_model.joblib", and main.py loads it at
# import time -- if it's missing the process dies on boot, before any request.
COPY model/ ./model/

# Run as an unprivileged user rather than root. Everything above is already
# written, and nothing at runtime writes to disk.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Documentation for `docker run -P` and human readers; publishing a port still
# requires an explicit -p on `docker run`.
EXPOSE 8000

# Shell form (no JSON array) so ${PORT:-8000} is expanded by /bin/sh at start:
# Render injects $PORT, local `docker run` doesn't and falls back to 8000.
# `exec` makes uvicorn REPLACE the shell as PID 1 instead of running as its
# child, so SIGTERM from `docker stop` / a Render redeploy reaches uvicorn and it
# shuts down gracefully rather than being killed after the 10s timeout.
# --host 0.0.0.0 is required -- uvicorn's 127.0.0.1 default is unreachable from
# outside the container. No --reload: that's a dev-only flag.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
