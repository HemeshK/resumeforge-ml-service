"""Derive score thresholds for the 3-way serving label (Good Fit / Potential
Fit / No Fit) from the saved binary model.

match_model.joblib was trained on a binary collapse (Good Fit=1, else=0), so
its output is P(Good Fit). To surface the original 3-class label at serve
time we need two cut points on that probability. This script scores the
same held-out same-domain test set used in RESULTS.md (never seen by the
saved model in training) and reports the score distribution per *original*
3-class label, so the thresholds are picked from real data rather than
guessed.

Usage:
    source venv/bin/activate
    python training/calibrate_thresholds.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import data
import features

MODEL_PATH = Path(__file__).parent.parent / "model" / "match_model.joblib"


def main():
    artifact = joblib.load(MODEL_PATH)
    model, vectorizer, feature_columns = artifact["model"], artifact["vectorizer"], artifact["feature_columns"]

    df = data.load_and_prepare()
    _, full_test = data.train_test_split_stratified(df)
    same_domain_test = data.same_domain_only(full_test)

    X, _ = features.build_features(same_domain_test, vectorizer)
    scores = model.predict_proba(X[feature_columns])[:, 1]
    labels = same_domain_test["label"].values

    print(f"n={len(scores)}\n")
    print(f"{'label':15s} {'n':>6s} {'p25':>8s} {'p50':>8s} {'p75':>8s} {'mean':>8s}")
    stats = {}
    for lbl in ["No Fit", "Potential Fit", "Good Fit"]:
        s = scores[labels == lbl]
        p25, p50, p75 = np.percentile(s, [25, 50, 75])
        stats[lbl] = p50
        print(f"{lbl:15s} {len(s):>6d} {p25:>8.4f} {p50:>8.4f} {p75:>8.4f} {s.mean():>8.4f}")

    potential_threshold = (stats["No Fit"] + stats["Potential Fit"]) / 2
    good_threshold = (stats["Potential Fit"] + stats["Good Fit"]) / 2

    print(f"\nRecommended thresholds (midpoint of medians):")
    print(f"  potential_threshold = {potential_threshold:.4f}  (score >= this -> at least 'Potential Fit')")
    print(f"  good_threshold      = {good_threshold:.4f}  (score >= this -> 'Good Fit')")


if __name__ == "__main__":
    main()
