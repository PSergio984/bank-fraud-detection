# Bank Account Fraud Scoring — Midterm Project Design

## Intent and why

ML Zoomcamp midterm project (submit Nov 17, peer review Nov 24). Goal beyond
the rubric: a portfolio piece that reads like production ML — fraud detection
is one of the most-requested skills in fintech job postings, and a 1%-positive
imbalanced problem forces real modeling work (PR curves, class weights,
threshold policy) instead of another 0.9-AUC toy.

## Problem

Score a bank account-opening application with its probability of being
fraudulent (`fraud_bool`). A bank's onboarding system calls the API at
application time; high scores route to manual review / step-up verification,
low scores auto-approve.

## Dataset

- **Bank Account Fraud (BAF), NeurIPS 2022** (Jesus et al.), `Base` variant —
  1M rows, 30 features, ~1.15% fraud rate. Public on Kaggle:
  `sgpjesus/bank-account-fraud-dataset-neurips-2022` (CC-BY-NC-SA).
- Semi-synthetic: generated to match a real bank's application distribution;
  disclosed openly in README. Not a course dataset — allowed.
- Features are interpretable: velocity (6h/24h/4w), address history, email/name
  similarity, credit risk score, device/session signals, employment/housing.
- `-1` is a missing-value sentinel in 6 columns → EDA/preprocessing handles it.
- ~400 MB CSV is NOT committed; README + `get_data.py` document the exact
  `kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022`
  command (reproducibility criterion satisfied). Kaggle CLI auth already
  exists on this machine (`~/.kaggle/access_token`); README also gives a
  manual browser-download fallback for reviewers without Kaggle creds.

## Goals (= rubric, 16 pts)

| Criterion | Plan |
|---|---|
| Problem description (2) | README: fraud cost, how the score is consumed, threshold policy |
| EDA (2) | distributions, missing/sentinels, class imbalance, correlations, mutual information + model-based feature importance |
| Model training (3) | logistic regression, random forest, XGBoost — each tuned against the validation split (notebook, subsample), compared on test |
| Notebook→script (1) | `train.py` trains final model + saves `model.bin` |
| Reproducibility (1) | pinned deps, data download steps, seeds fixed |
| Deployment (1) | FastAPI service: `POST /predict`, `GET /health` |
| Dependency mgmt (2) | `requirements.txt` + venv instructions in README |
| Containerization (2) | `Dockerfile` + build/run commands in README |
| Cloud (0–2) | attempt Render/Railway free tier; fall back to screenshot/video |

## Non-goals

- No streaming/real-time feature store — single REST call, stateless.
- No deep fairness audit (dataset was built for fairness research; README notes
  it as future work, not a deliverable).
- No custom frontend — curl/PowerShell request examples as evidence.
- No multi-variant BAF comparison — `Base` only.

## Constraints

- Windows dev box, Docker Desktop present, Python 3.13.
- Iterate on a stratified subsample (~100–200k) for notebook speed; final
  `train.py` trains on the full 1M (30 features — feasible locally).
- Deadline Nov 17; Docker + deploy last-week, not last-day.

## Approaches considered

1. **FastAPI + uvicorn, scikit-learn + XGBoost, pickled model** — *chosen.*
   Modern resume signal, single `model.bin` artifact, trivially Dockerizable.
2. Flask + gunicorn — course-default; works, but FastAPI shows async/Pydantic
   skills and still satisfies "Flask or similar framework."
3. BentoML — overkill; adds a framework reviewers must learn for zero rubric gain.

## The how (decided)

- **Metric story:** ROC-AUC + PR-AUC + recall at chosen threshold + cost
  framing. Threshold policy (builder check decision): review capacity is the
  business constraint — pick the threshold with **max recall subject to
  flag-rate ≤ 5% on validation**, report metrics on test; also report the
  F1-optimal threshold for comparison.
- **Split:** 60/20/20 stratified train/val/test. Secondary temporal check in
  the notebook: train months ≤5, test months 6–7 (the BAF paper's protocol —
  guards against the random split looking inflated).
- **Tuning scope:** hyperparameter search happens in the notebook on a
  stratified subsample (~200k rows) against a validation split. `train.py`
  does NOT search — it refits the chosen config on train+val of the full 1M
  and evaluates on test.
- **Preprocessing:** -1 sentinels → NaN + `was_missing` indicators where useful;
  one-hot for low-cardinality categoricals (`payment_type`,
  `employment_status`, `housing_status`, `source`, `device_os`); numeric passthrough.
- **Models:** `DummyClassifier` baseline → tuned `LogisticRegression` (C) →
  tuned `RandomForest` (depth, n_estimators, class_weight) → tuned `XGBoost`
  (eta, max_depth, scale_pos_weight). Pick best on val PR-AUC, tie-break on
  recall@cap.
- **Artifacts:** `model.bin` = dict(model, fitted preprocessor, threshold,
  feature list). Hard budget: <100 MB (GitHub file limit + free-tier RAM) —
  enforced via bounded depth/estimators; XGBoost artifacts stay small anyway.
- **Service:** `predict.py` FastAPI app; request = JSON of the 30 fields;
  response = `{fraud_score, decision}` where decision = score ≥ threshold.
  Score is rank-consistent, NOT a calibrated probability (class weighting
  shifts outputs); README states this. Optional stretch: isotonic calibration
  on validation, then `fraud_score` becomes interpretable as a probability.
- **Repo:** `README.md`, `notebook.ipynb`, `train.py`, `predict.py`,
  `test_predict.py` (sanity request), `get_data.py` (kaggle download),
  `requirements.txt`, `Dockerfile`, `docs/`.

## Open / left to builder

- Exact hyperparameter grids (keep small: 3–6 values each).
- Whether `device_fraud_count`/`month` survive leakage review — verify in EDA.
- Screenshot vs short screen-recording as the deploy evidence.
- Optional stretch: a second endpoint `/explain` returning top-k feature
  contributions (cheap "noticed" extra) — only if time allows.
