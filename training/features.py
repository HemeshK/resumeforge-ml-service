"""Feature engineering for resume<->JD match scoring.

Every feature is derived from `resume`/`jd` text alone. `resume_domain` /
`jd_domain` are never used as model inputs -- that's the whole point of the
ablation in train.py.

Feature set:
  - tfidf_cosine    : cosine similarity between TF-IDF vectors of resume & jd
  - keyword_overlap : symmetric Jaccard overlap of non-stopword tokens
  - jd_coverage     : fraction of the JD's top TF-IDF terms that also appear
                       in the resume (asymmetric -- "did the resume cover
                       what the JD asked for")
  - exp_gap         : resume years-of-experience minus JD years-required,
                       via regex (0 when either side can't be parsed)
  - exp_missing     : 1 if years couldn't be parsed from one/both sides
  - resume_len      : resume word count
  - jd_len          : JD word count
  - len_ratio       : resume_len / jd_len
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z+#.]{2,}")
YEARS_RE = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
STOP_WORDS = ENGLISH_STOP_WORDS

TFIDF_KWARGS = dict(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    sublinear_tf=True,
)

JD_KEYWORD_TOPK = 15

FEATURE_COLUMNS = [
    "tfidf_cosine",
    "keyword_overlap",
    "jd_coverage",
    "exp_gap",
    "exp_missing",
    "resume_len",
    "jd_len",
    "len_ratio",
]


def fit_vectorizer(train_df: pd.DataFrame) -> TfidfVectorizer:
    corpus = pd.concat([train_df["resume"], train_df["jd"]], ignore_index=True).fillna("")
    vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
    vectorizer.fit(corpus)
    return vectorizer


def _tokenize(text: str) -> set:
    return {t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOP_WORDS}


def _extract_years(text: str):
    matches = YEARS_RE.findall(text or "")
    nums = [int(m) for m in matches if int(m) <= 50]
    return max(nums) if nums else None


def _cosine_rowwise(a_sparse, b_sparse) -> np.ndarray:
    dot = np.asarray(a_sparse.multiply(b_sparse).sum(axis=1)).ravel()
    a_norm = np.sqrt(np.asarray(a_sparse.multiply(a_sparse).sum(axis=1)).ravel())
    b_norm = np.sqrt(np.asarray(b_sparse.multiply(b_sparse).sum(axis=1)).ravel())
    denom = a_norm * b_norm
    return np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)


def _jd_coverage_batch(jd_tfidf, resume_tokens_list, feature_names, topk=JD_KEYWORD_TOPK) -> np.ndarray:
    jd_csr = jd_tfidf.tocsr()
    out = np.zeros(jd_csr.shape[0])
    for i in range(jd_csr.shape[0]):
        start, end = jd_csr.indptr[i], jd_csr.indptr[i + 1]
        idx = jd_csr.indices[start:end]
        data = jd_csr.data[start:end]
        if len(idx) == 0:
            continue
        top_order = np.argsort(data)[-topk:]
        top_terms = set()
        for fi in idx[top_order]:
            top_terms.update(feature_names[fi].split())
        if not top_terms:
            continue
        out[i] = len(top_terms & resume_tokens_list[i]) / len(top_terms)
    return out


def build_features(df: pd.DataFrame, vectorizer: TfidfVectorizer):
    """Returns (features_df, y) both positionally aligned to a fresh 0..n-1 index."""
    df = df.reset_index(drop=True)
    resumes = df["resume"].fillna("")
    jds = df["jd"].fillna("")

    resume_tfidf = vectorizer.transform(resumes)
    jd_tfidf = vectorizer.transform(jds)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_cosine = _cosine_rowwise(resume_tfidf, jd_tfidf)

    resume_tokens_list = [_tokenize(t) for t in resumes]
    jd_tokens_list = [_tokenize(t) for t in jds]

    keyword_overlap = np.array(
        [
            len(r & j) / len(r | j) if (r | j) else 0.0
            for r, j in zip(resume_tokens_list, jd_tokens_list)
        ]
    )

    jd_coverage = _jd_coverage_batch(jd_tfidf, resume_tokens_list, feature_names)

    resume_years = [_extract_years(t) for t in resumes]
    jd_years = [_extract_years(t) for t in jds]
    exp_gap = np.array(
        [
            (ry - jy) if (ry is not None and jy is not None) else 0.0
            for ry, jy in zip(resume_years, jd_years)
        ]
    )
    exp_missing = np.array(
        [1 if (ry is None or jy is None) else 0 for ry, jy in zip(resume_years, jd_years)]
    )

    resume_len = np.array([len(t.split()) for t in resumes])
    jd_len = np.array([len(t.split()) for t in jds])
    len_ratio = resume_len / np.maximum(jd_len, 1)

    feat_df = pd.DataFrame(
        {
            "tfidf_cosine": tfidf_cosine,
            "keyword_overlap": keyword_overlap,
            "jd_coverage": jd_coverage,
            "exp_gap": exp_gap,
            "exp_missing": exp_missing,
            "resume_len": resume_len,
            "jd_len": jd_len,
            "len_ratio": len_ratio,
        }
    )[FEATURE_COLUMNS]

    return feat_df, df["label_bin"].values
