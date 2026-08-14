"""Train the resume<->JD match-score model.

Key experiment (the ablation): does training on cross-domain pairs (e.g. a
finance resume vs. a software JD) inflate reported metrics via a trivial
"domains obviously don't match" shortcut? We train two classifiers and
evaluate BOTH on the same same-domain-only test set:

  Model A - trained on ALL training data (same-domain + cross-domain)
  Model B - trained on SAME-DOMAIN training data only

If Model A's advantage evaporates (or reverses) once cross-domain rows are
gone from the training set, that's evidence the full-data model was leaning
on the domain-shortcut rather than genuine content matching.

Usage:
    source venv/bin/activate
    python training/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
import data
import features

MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "match_model.joblib"
RESULTS_PATH = Path(__file__).parent / "RESULTS.md"
METRICS_PATH = Path(__file__).parent / "metrics.json"

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

THRESHOLD = 0.5


def make_model(y_train: np.ndarray) -> XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)


def evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
    }


def fmt_cm(cm) -> str:
    (tn, fp), (fn, tp) = cm
    return (
        f"              predicted 0   predicted 1\n"
        f"  actual 0    {tn:>10}   {fp:>11}\n"
        f"  actual 1    {fn:>10}   {tp:>11}"
    )


def print_metrics(name: str, m: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"n={m['n']}  positives={m['n_pos']} ({m['n_pos'] / m['n']:.2%})")
    print(f"ROC-AUC:  {m['auc']:.4f}")
    print(f"Macro-F1: {m['macro_f1']:.4f}")
    print(fmt_cm(m["confusion_matrix"]))


def write_results_md(ctx: dict) -> None:
    a, b, baseline = ctx["model_a"], ctx["model_b"], ctx["baseline"]
    winner = ctx["winner"]

    def row(name, m):
        return f"| {name} | {m['auc']:.4f} | {m['macro_f1']:.4f} |"

    md = f"""# Match-score model: ablation results

## What this tests

Does mixing cross-domain resume/JD pairs (e.g. a finance resume paired with
a software JD) into training inflate metrics through a trivial domain
shortcut, rather than the model actually learning content match quality?

- **Model A** — XGBoost trained on ALL training data (same-domain + cross-domain pairs).
- **Model B** — XGBoost trained on SAME-DOMAIN-ONLY training data.
- Both are evaluated on the **same held-out same-domain-only test set**
  ({baseline['n']} rows, {baseline['n_pos']} positive / {baseline['n_pos'] / baseline['n']:.1%}).
  `resume_domain` / `jd_domain` are never used as model features — only for
  slicing which rows go into training/eval.
- The TF-IDF vectorizer is fit once on the full training corpus and shared
  by both models, so the only variable being ablated is *which rows the
  classifier trains on*, not the text representation.

## Dataset

- Source: `med2425/resume-job-fit-merged-v1` (Hugging Face), train+test splits pooled and re-split.
- {ctx['total_rows']} total rows. Label collapsed to binary: `Good Fit` = 1, `Potential Fit`/`No Fit` = 0.
- Positive rate: {ctx['pos_rate']:.2%} (~{ctx['imbalance_ratio']:.1f}:1 imbalance) — handled via stratified splitting and XGBoost `scale_pos_weight`.
- {ctx['same_domain_frac']:.1%} of all rows are same-domain (`resume_domain == jd_domain`).
- Train rows: {ctx['n_full_train']} full / {ctx['n_same_train']} same-domain-only.
- Test rows: {ctx['n_full_test']} full / {ctx['n_same_test']} same-domain-only (the eval set used below).

## Features

Derived from `resume`/`jd` text only — no domain columns:

- `tfidf_cosine` — cosine similarity between TF-IDF vectors of resume and JD
- `keyword_overlap` — symmetric Jaccard overlap of non-stopword tokens
- `jd_coverage` — fraction of the JD's top TF-IDF terms also present in the resume
- `exp_gap` / `exp_missing` — regex-extracted years-of-experience gap (resume − JD required)
- `resume_len` / `jd_len` / `len_ratio` — word-count features

## Results (all evaluated on the same-domain-only test set)

| Model | ROC-AUC | Macro-F1 |
|---|---|---|
| Floor baseline (raw TF-IDF cosine, no model) | {baseline['auc']:.4f} | — |
{row('Model A (trained on ALL data)', a)}
{row('Model B (trained on SAME-DOMAIN data only)', b)}

**Model A confusion matrix** (n={a['n']}):
```
{fmt_cm(a['confusion_matrix'])}
```

**Model B confusion matrix** (n={b['n']}):
```
{fmt_cm(b['confusion_matrix'])}
```

## Plain-English summary

The floor baseline — just thresholding raw TF-IDF cosine similarity, no
model at all — already gets {baseline['auc']:.4f} ROC-AUC on the same-domain
test set. That's the bar any trained model needs to clear to be worth its
complexity.

{ctx['narrative']}

**Model saved:** {winner['label']} ({'trained on all data' if winner['key'] == 'model_a' else 'trained on same-domain data only'}),
chosen because it scored higher on macro-F1 ({winner['metrics']['macro_f1']:.4f} vs.
{ctx['loser']['metrics']['macro_f1']:.4f}) on the same-domain-only test set — the
metric that reflects real classification quality on the harder,
non-shortcuttable slice of the data. Saved to `model/match_model.joblib`
alongside the fitted TF-IDF vectorizer.

## Caveats

- Threshold fixed at 0.5 for macro-F1 / confusion matrices — not tuned per model.
- `exp_gap` regex only catches "N years/yrs" phrasing; resumes/JDs that state
  experience differently (date ranges only, "senior", etc.) fall back to
  `exp_missing=1`, `exp_gap=0`.
- This ablation isolates *training-data domain composition*. It says nothing
  about whether the features themselves are the best possible ones.
"""
    RESULTS_PATH.write_text(md)


def main():
    print("Loading dataset...")
    df = data.load_and_prepare()
    pos_rate = df["label_bin"].mean()
    print(f"Total rows: {len(df)} | Good Fit rate: {pos_rate:.2%} (~{(1 - pos_rate) / pos_rate:.1f}:1 imbalance)")

    full_train, full_test = data.train_test_split_stratified(df)
    same_domain_train = data.same_domain_only(full_train)
    same_domain_test = data.same_domain_only(full_test)
    same_domain_frac = (df["resume_domain"] == df["jd_domain"]).mean()

    print(f"full_train={len(full_train)}  full_test={len(full_test)}")
    print(f"same_domain_train={len(same_domain_train)}  same_domain_test={len(same_domain_test)}")

    print("\nFitting TF-IDF vectorizer on full_train corpus (shared by both models)...")
    vectorizer = features.fit_vectorizer(full_train)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\nBuilding features...")
    X_full_train, y_full_train = features.build_features(full_train, vectorizer)
    X_same_train, y_same_train = features.build_features(same_domain_train, vectorizer)
    X_same_test, y_same_test = features.build_features(same_domain_test, vectorizer)

    print("\nFloor baseline: raw TF-IDF cosine similarity, no model, no other features.")
    baseline_auc = roc_auc_score(y_same_test, X_same_test["tfidf_cosine"].values)
    print(f"Baseline ROC-AUC (tfidf_cosine only): {baseline_auc:.4f}")
    baseline = {"auc": baseline_auc, "n": len(y_same_test), "n_pos": int(y_same_test.sum())}

    print("\nTraining Model A (ALL training data)...")
    model_a = make_model(y_full_train)
    model_a.fit(X_full_train, y_full_train)
    proba_a = model_a.predict_proba(X_same_test)[:, 1]
    metrics_a = evaluate(y_same_test, proba_a)
    print_metrics("Model A (trained on ALL data) -> same-domain test", metrics_a)

    print("\nTraining Model B (SAME-DOMAIN training data only)...")
    model_b = make_model(y_same_train)
    model_b.fit(X_same_train, y_same_train)
    proba_b = model_b.predict_proba(X_same_test)[:, 1]
    metrics_b = evaluate(y_same_test, proba_b)
    print_metrics("Model B (trained on SAME-DOMAIN data only) -> same-domain test", metrics_b)

    print("\n=== Final metrics table (same-domain-only test set) ===")
    header = f"{'Model':45s} {'ROC-AUC':>10s} {'Macro-F1':>10s}"
    print(header)
    print("-" * len(header))
    print(f"{'Floor baseline (TF-IDF cosine only)':45s} {baseline['auc']:>10.4f} {'--':>10s}")
    print(f"{'Model A (all data)':45s} {metrics_a['auc']:>10.4f} {metrics_a['macro_f1']:>10.4f}")
    print(f"{'Model B (same-domain only)':45s} {metrics_b['auc']:>10.4f} {metrics_b['macro_f1']:>10.4f}")

    if metrics_b["macro_f1"] > metrics_a["macro_f1"]:
        winner_key, winner_model, winner_metrics = "model_b", model_b, metrics_b
        loser_key, loser_metrics = "model_a", metrics_a
        narrative = (
            "Model B (same-domain-only training) wins on the same-domain test set, "
            f"macro-F1 {metrics_b['macro_f1']:.4f} vs. Model A's {metrics_a['macro_f1']:.4f} "
            f"(ROC-AUC {metrics_b['auc']:.4f} vs. {metrics_a['auc']:.4f}). This supports the "
            "hypothesis: training on the full mix, including easy cross-domain negatives, "
            "doesn't help the model separate genuinely close same-domain candidates — if "
            "anything it dilutes training signal on the harder distinction the deployed "
            "model actually needs to make."
        )
    else:
        winner_key, winner_model, winner_metrics = "model_a", model_a, metrics_a
        loser_key, loser_metrics = "model_b", metrics_b
        narrative = (
            "Model A (trained on all data, including cross-domain pairs) still wins on the "
            f"same-domain test set, macro-F1 {metrics_a['macro_f1']:.4f} vs. Model B's "
            f"{metrics_b['macro_f1']:.4f} (ROC-AUC {metrics_a['auc']:.4f} vs. {metrics_b['auc']:.4f}). "
            "The extra cross-domain training rows evidently still contribute useful signal "
            "(more training data, same feature space) beyond a pure domain shortcut — though "
            "note the same-domain-only Model B was trained on far fewer rows, which likely "
            "also holds it back."
        )

    print(f"\nSaving winner ({winner_key}) + vectorizer to {MODEL_PATH} ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": winner_model,
            "vectorizer": vectorizer,
            "feature_columns": features.FEATURE_COLUMNS,
            "threshold": THRESHOLD,
            "trained_on": winner_key,
        },
        MODEL_PATH,
    )
    print("Saved.")

    ctx = {
        "total_rows": len(df),
        "pos_rate": pos_rate,
        "imbalance_ratio": (1 - pos_rate) / pos_rate,
        "same_domain_frac": same_domain_frac,
        "n_full_train": len(full_train),
        "n_same_train": len(same_domain_train),
        "n_full_test": len(full_test),
        "n_same_test": len(same_domain_test),
        "baseline": baseline,
        "model_a": metrics_a,
        "model_b": metrics_b,
        "narrative": narrative,
        "winner": {"key": winner_key, "label": "Model A" if winner_key == "model_a" else "Model B", "metrics": winner_metrics},
        "loser": {"key": loser_key, "metrics": loser_metrics},
    }
    write_results_md(ctx)
    print(f"Wrote {RESULTS_PATH}")

    # Machine-readable summary for CI (the regression gate reads winner.auc).
    metrics = {
        "baseline_auc": baseline["auc"],
        "model_a": {"auc": metrics_a["auc"], "macro_f1": metrics_a["macro_f1"]},
        "model_b": {"auc": metrics_b["auc"], "macro_f1": metrics_b["macro_f1"]},
        "winner": {"key": winner_key, "auc": winner_metrics["auc"], "macro_f1": winner_metrics["macro_f1"]},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
