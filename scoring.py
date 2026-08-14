"""Resume<->JD match scoring: loads model/match_model.joblib and scores a
single (resume, jd) pair.

Feature logic mirrors training/features.py, reimplemented here for a single
pair instead of a batch DataFrame so the service doesn't need pandas or the
training/ package (training/ is not deployed -- see CLAUDE.md). Keep the two
in sync if the feature set changes.
"""
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_PATH = Path(__file__).parent / "model" / "match_model.joblib"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z+#.]{2,}")
YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
STOP_WORDS = ENGLISH_STOP_WORDS
JD_KEYWORD_TOPK = 15

# Derived in training/calibrate_thresholds.py from the score distribution of
# each original 3-class label on the held-out same-domain test set.
POTENTIAL_FIT_THRESHOLD = 0.2590
GOOD_FIT_THRESHOLD = 0.5956


def load_artifact(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


def _tokenize(text: str) -> set:
    return {t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOP_WORDS}


def _extract_years(text: str):
    matches = YEARS_RE.findall(text or "")
    nums = [int(m) for m in matches if int(m) <= 50]
    return max(nums) if nums else None


def compute_features(resume: str, jd: str, vectorizer) -> dict:
    resume_vec = vectorizer.transform([resume or ""])
    jd_vec = vectorizer.transform([jd or ""])
    feature_names = vectorizer.get_feature_names_out()

    dot = resume_vec.multiply(jd_vec).sum()
    resume_norm = np.sqrt(resume_vec.multiply(resume_vec).sum())
    jd_norm = np.sqrt(jd_vec.multiply(jd_vec).sum())
    denom = resume_norm * jd_norm
    tfidf_cosine = float(dot / denom) if denom > 0 else 0.0

    resume_tokens = _tokenize(resume)
    jd_tokens = _tokenize(jd)
    union = resume_tokens | jd_tokens
    keyword_overlap = len(resume_tokens & jd_tokens) / len(union) if union else 0.0

    jd_csr = jd_vec.tocsr()
    top_terms = set()
    if jd_csr.nnz > 0:
        idx, values = jd_csr.indices, jd_csr.data
        top_order = np.argsort(values)[-JD_KEYWORD_TOPK:]
        for fi in idx[top_order]:
            top_terms.update(feature_names[fi].split())
    jd_coverage = len(top_terms & resume_tokens) / len(top_terms) if top_terms else 0.0

    resume_years = _extract_years(resume)
    jd_years = _extract_years(jd)
    if resume_years is not None and jd_years is not None:
        exp_gap = float(resume_years - jd_years)
        exp_missing = 0
    else:
        exp_gap = 0.0
        exp_missing = 1

    resume_len = len((resume or "").split())
    jd_len = len((jd or "").split())
    len_ratio = resume_len / max(jd_len, 1)

    return {
        "tfidf_cosine": tfidf_cosine,
        "keyword_overlap": keyword_overlap,
        "jd_coverage": jd_coverage,
        "exp_gap": exp_gap,
        "exp_missing": exp_missing,
        "resume_len": float(resume_len),
        "jd_len": float(jd_len),
        "len_ratio": len_ratio,
    }


def _label_for_score(score: float) -> str:
    if score >= GOOD_FIT_THRESHOLD:
        return "Good Fit"
    if score >= POTENTIAL_FIT_THRESHOLD:
        return "Potential Fit"
    return "No Fit"


def score_pair(resume: str, jd: str, artifact: dict) -> dict:
    model, vectorizer, feature_columns = (
        artifact["model"],
        artifact["vectorizer"],
        artifact["feature_columns"],
    )
    feature_values = compute_features(resume, jd, vectorizer)
    ordered = np.array([[feature_values[col] for col in feature_columns]])
    score = float(model.predict_proba(ordered)[0, 1])
    return {
        "score": score,
        "label": _label_for_score(score),
        "features": feature_values,
    }
