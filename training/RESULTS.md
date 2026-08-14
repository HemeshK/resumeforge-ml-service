# Match-score model: ablation results

## What this tests

Does mixing cross-domain resume/JD pairs (e.g. a finance resume paired with
a software JD) into training inflate metrics through a trivial domain
shortcut, rather than the model actually learning content match quality?

- **Model A** — XGBoost trained on ALL training data (same-domain + cross-domain pairs).
- **Model B** — XGBoost trained on SAME-DOMAIN-ONLY training data.
- Both are evaluated on the **same held-out same-domain-only test set**
  (10195 rows, 1284 positive / 12.6%).
  `resume_domain` / `jd_domain` are never used as model features — only for
  slicing which rows go into training/eval.
- The TF-IDF vectorizer is fit once on the full training corpus and shared
  by both models, so the only variable being ablated is *which rows the
  classifier trains on*, not the text representation.

## Dataset

- Source: `med2425/resume-job-fit-merged-v1` (Hugging Face), train+test splits pooled and re-split.
- 93733 total rows. Label collapsed to binary: `Good Fit` = 1, `Potential Fit`/`No Fit` = 0.
- Positive rate: 7.97% (~11.5:1 imbalance) — handled via stratified splitting and XGBoost `scale_pos_weight`.
- 54.3% of all rows are same-domain (`resume_domain == jd_domain`).
- Train rows: 74986 full / 40705 same-domain-only.
- Test rows: 18747 full / 10195 same-domain-only (the eval set used below).

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
| Floor baseline (raw TF-IDF cosine, no model) | 0.8137 | — |
| Model A (trained on ALL data) | 0.9079 | 0.6900 |
| Model B (trained on SAME-DOMAIN data only) | 0.9212 | 0.7279 |

**Model A confusion matrix** (n=10195):
```
              predicted 0   predicted 1
  actual 0          6943          1968
  actual 1           162          1122
```

**Model B confusion matrix** (n=10195):
```
              predicted 0   predicted 1
  actual 0          7378          1533
  actual 1           188          1096
```

## Plain-English summary

The floor baseline — just thresholding raw TF-IDF cosine similarity, no
model at all — already gets 0.8137 ROC-AUC on the same-domain
test set. That's the bar any trained model needs to clear to be worth its
complexity.

Model B (same-domain-only training) wins on the same-domain test set, macro-F1 0.7279 vs. Model A's 0.6900 (ROC-AUC 0.9212 vs. 0.9079). This supports the hypothesis: training on the full mix, including easy cross-domain negatives, doesn't help the model separate genuinely close same-domain candidates — if anything it dilutes training signal on the harder distinction the deployed model actually needs to make.

**Model saved:** Model B (trained on same-domain data only),
chosen because it scored higher on macro-F1 (0.7279 vs.
0.6900) on the same-domain-only test set — the
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
