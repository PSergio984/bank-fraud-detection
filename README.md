# Bank Account Fraud Detection

ML Zoomcamp 2026 midterm project — scores new bank account applications for
fraud risk and flags the riskiest ones for manual review.

## The problem

Online account opening is a prime fraud vector: a fraudster who gets an account
approved can run first-party fraud, money mule activity, or bust-out credit
schemes. Real banks cannot manually review every application — a review team
has a fixed capacity. The useful ML question is therefore not "is this fraud,
yes/no" but **"which applications should the review team look at first?"**

This project trains a model that scores each application with a fraud-risk
score and deploys it as an HTTP service. The service applies a threshold
chosen so that **at most ~5% of applications are flagged** — matching a
hypothetical review team's capacity. Under that cap the model catches about
**53% of all fraud** (vs. the 1.15% fraud base rate — a ~46x lift over random
triage). It is a prioritization tool, not an auto-reject system: a human
reviewer makes the final call.

## Dataset

- **Source:** [Bank Account Fraud (BAF) suite](https://github.com/feedzai/bank-account-fraud),
  published at NeurIPS 2022 by Feedzai (Jesus et al., *"Turning the Tables:
  Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation"*).
- **Variant used:** `Base.csv` — 1,000,000 applications, 32 columns
  (30 predictors + `fraud_bool` target + `month`).
- **License:** CC BY-NC-SA 4.0 — semi-synthetic data generated to mirror real
  fraud distributions while protecting customer privacy.
- **Target:** `fraud_bool` — 1.15% positive (severe class imbalance, so
  accuracy is meaningless; we use ROC-AUC, PR-AUC, and recall-at-capacity).
- **Download:** `python get_data.py` (uses the Kaggle API — needs
  `~/.kaggle/kaggle.json`), or download `Base.csv` manually from
  [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
  → place it in `data/`.

## Exploratory data analysis (see `notebook.ipynb`)

- **Sentinel missingness is a fraud signal.** Six features use `-1` as a
  missing sentinel. `prev_address_months_count` is missing for 71% of rows —
  and rows where it is missing are **4.6x more fraudulent** (1.42%) than rows
  where it is present (0.31%). Same pattern for `bank_months_count`
  (25% missing; 1.63% vs 0.92%). The pipeline therefore adds
  `*_was_missing` indicators instead of dropping these columns.
- **Top predictive signals** (mutual information + permutation importance):
  address-history length, `keep_alive_session`, phone/email validity,
  `email_is_free`, employment status.
- **`device_fraud_count`** is kept as a feature but flagged in the notebook:
  it aggregates prior fraud on the device and risks near-leakage; the
  temporal holdout check (below) verifies the model generalizes without it
  dominating.
- **Temporal drift exists:** fraud rate rises from ~1.0% (months 0–5) to 1.4%
  (months 6–7) — evaluated explicitly rather than assumed away.

## Models compared (validation split, 200k stratified subsample)

| model | ROC-AUC | PR-AUC | recall @ 5% cap |
|---|---|---|---|
| **XGBoost d=4, lr=0.05** | **0.8745** | **0.1225** | **0.4882** |
| XGBoost d=4, lr=0.1 | 0.8664 | 0.1119 | 0.4763 |
| XGBoost d=6, lr=0.05 | 0.8615 | 0.1066 | 0.4573 |
| Logistic regression (C tuned) | 0.8698 | 0.1061 | 0.4668 |
| XGBoost d=6, lr=0.1 | 0.8389 | 0.0912 | 0.4194 |
| Random forest (depth/trees tuned) | 0.8582 | 0.0843 | 0.4265 |
| Dummy baseline | 0.5000 | 0.0106 | 0.0521 |

Winner selected by validation **PR-AUC** (the right metric under 1% fraud),
with recall-at-capacity as tie-breaker. XGBoost also beat logistic regression
on ROC-AUC — the nonlinear interactions (e.g., missingness × behavior)
matter here.

## Final model — held-out test results

`train.py` refits the winning config on train+validation (full 1M rows),
picks the threshold for max recall under the 5% flag-rate cap on validation,
and evaluates once on the untouched test split:

| metric | value |
|---|---|
| ROC-AUC | **0.897** |
| PR-AUC | **0.177** |
| Flag rate (review capacity) | 5.05% |
| Recall at cap (fraud caught) | **53.0%** |
| Precision at cap | 11.6% |
| Threshold | 0.7876 |

**Temporal holdout** (train months ≤5 → test months 6–7): ROC-AUC 0.881,
PR-AUC 0.165, recall at cap 48.6% — consistent with the random split, so the
metrics are not inflated by time leakage.

For comparison, the alternative "best-F1" threshold would flag only 1.7% of
applications but catch just 27.5% of fraud — the capacity-driven threshold is
the operationally meaningful one.

## Project structure

```
notebook.ipynb        EDA, feature importance, model tuning, threshold & temporal analysis
get_data.py           downloads data/Base.csv via the Kaggle API
features.py           shared load/clean/split pipeline (train and serve use the same code)
train.py              trains the final model -> model.bin
predict.py            FastAPI service: GET /health, POST /predict
test_*.py             pytest suite (features, artifact contract, API edge cases)
requirements.txt      runtime deps (pinned) — what Docker installs
requirements-dev.txt  + notebook/test tooling
Dockerfile            python:3.12-slim serving image
model.bin             trained artifact (committed for one-command deploy; regenerate with train.py)
data/                 gitignored — 700MB CSV, fetch via get_data.py
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use source .venv/bin/activate on *nix
pip install -r requirements-dev.txt

python get_data.py              # fetch data/Base.csv (needs Kaggle API key)

python train.py                 # full 1M-row training -> model.bin  (~10-15 min)
# or a fast dev run on 50k rows:
#   TRAIN_SAMPLE=50000 python train.py

uvicorn predict:app --host 0.0.0.0 --port 9696
```

Tests: `pytest` (13 tests: preprocessing, artifact contract, API edge cases).

## Run with Docker

```bash
docker build -t bank-fraud .    # model.bin is committed — no training needed
docker run --rm -p 9696:9696 bank-fraud
```

(To rebuild from a fresh model, run `python train.py` first — it overwrites
`model.bin`.)

Verified: image builds clean (1.5 GB — most of it is xgboost's Linux NCCL
dependency), `/health` and `/predict` both respond inside the container.

## API

`GET /health` → `{"status": "ok"}`

`POST /predict` — body: all 30 application features as JSON.
`POST /predict` →

```json
{"fraud_score": 0.837, "decision": true}
```

`fraud_score` is the model's risk score (higher = riskier); `decision` is
`true` when the score exceeds the stored 5%-capacity threshold, i.e. "route to
manual review". Note the score is a ranking signal, not a calibrated
probability — see limitations.

Missing required fields → HTTP 422. Unknown categorical values and extra
fields are tolerated (DictVectorizer semantics). `sample-request.json` in
this repo is a ready-to-use example body
(`curl -X POST localhost:9696/predict -H "Content-Type: application/json" -d @sample-request.json`);
the Swagger UI at `/docs` also shows the full schema.

## Deployment evidence

Local Docker verification (Task checklist evidence): see `screenshots/` —
container running, `/health` 200, and a `/predict` response. (Cloud deploy
tracked separately — the image is platform-agnostic: any host that runs a
container on port 9696 works.)

## Limitations & next steps

- **Score ≠ calibrated probability.** The model uses `scale_pos_weight` for
  class imbalance, which shifts raw outputs. For real use, fit isotonic
  calibration on validation and report true probabilities.
- **Precision is honest, not pretty.** At 11.6% precision, ~9 of 10 flagged
  applications are innocent — that is the real economics of 1%-base-rate
  fraud review, and why the model ranks rather than decides.
- **No fairness evaluation.** The BAF suite ships with protected-attribute
  variants (e.g., age groups) designed for fairness study — a natural next
  step before any real deployment.
- **Drift monitoring.** The temporal split already shows fraud-rate drift;
  a production version would monitor score distribution + flag rate over time.
