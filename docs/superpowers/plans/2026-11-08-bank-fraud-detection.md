# Bank Account Fraud Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Midterm project — train, evaluate, and deploy a bank-account-opening
fraud scorer as a containerized FastAPI service, covering the full zoomcamp
rubric (16 pts).

**Architecture:** Shared `features.py` module (load/clean/split) feeds an
executed `notebook.ipynb` (EDA + tuning on ~200k subsample) and `train.py`
(refit winner on train+val of the full 1M → `model.bin`). `predict.py` serves
`model.bin` via FastAPI; Docker + cloud attempt close it out.

**Tech Stack:** Python 3.13, pandas, scikit-learn, XGBoost, DictVectorizer
(zoomcamp-idiomatic), FastAPI + uvicorn, Docker.

**Spec:** `docs/superpowers/specs/2026-11-08-bank-fraud-detection-design.md`

## Global Constraints

- Dataset: `data/Base.csv` — BAF `Base` variant, 1M rows, 32 cols, target
  `fraud_bool` (~1.15%). NOT committed to git.
- Sentinel cols (-1 = missing): `prev_address_months_count`,
  `current_address_months_count`, `credit_risk_score`, `bank_months_count`,
  `session_length_in_minutes`, `device_distinct_emails_8w`.
- Categorical cols: `payment_type`, `employment_status`, `housing_status`,
  `source`, `device_os`. All other non-target cols are numeric.
- Split: 60/20/20 stratified, seed 42, indices produced ONLY by
  `features.make_splits`.
- Threshold: chosen on validation = max recall s.t. flag-rate ≤ 5%;
  F1-optimal reported alongside.
- `model.bin` < 100 MB. Random seed 42 everywhere.
- API contract: `POST /predict` → `{"fraud_score": float, "decision": bool}`;
  `GET /health` → `{"status": "ok"}`.

## Review Focus

Inputs the spec doesn't pin that would bite a user:

1. Request with missing fields → Pydantic 422 listing what's missing
   (test in Task 5).
2. Request containing `-1` sentinels → identical cleaning path as training,
   no crash (test in Task 5).
3. Unknown categorical value at predict time → DictVectorizer ignores it,
   still returns a score (test in Task 5).
4. Extra unexpected JSON fields → ignored, not an error (test in Task 5).
5. `model.bin` missing at service start → clear startup error naming the
   fix (`python train.py`) (asserted in Task 5).

---

### Task 1: Repo scaffold, venv, data

**Files:**
- Create: `.gitignore`, `.dockerignore`, `get_data.py`, `requirements.txt`
- Create: `data/` (gitignored)

**Interfaces:**
- Produces: `data/Base.csv` present locally; `get_data.py` runnable as
  `python get_data.py` → prints row count of `data/Base.csv`.

- [ ] **Step 1: Init git + ignore files**

`git init`. `.gitignore`: `data/`, `model.bin`, `.venv/`, `__pycache__/`,
`*.pyc`, `catboost_info/`, `.ipynb_checkpoints/`. `.dockerignore`: same plus
`.git/`, `docs/`, `notebook.ipynb`.

- [ ] **Step 2: Write `get_data.py`**

If `data/Base.csv` exists → print shape and exit. Else run
`kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data --unzip`
via `subprocess`; on `FileNotFoundError` (no kaggle CLI) print manual steps
(Kaggle → dataset page → Download → unzip `Base.csv` into `data/`).

- [ ] **Step 3: Stage the data + venv + requirements**

Copy `D:\ai-eng\_dataset-spike\data\bank_fraud\Base.csv` → `data/Base.csv`
(already downloaded; `get_data.py` is for reviewers). Verify:
`python get_data.py` prints `Base.csv shape: (1000000, 32)`.
Create `.venv`, activate, `pip install pandas numpy scikit-learn xgboost
fastapi "uvicorn[standard]" matplotlib seaborn jupyter pytest requests`.

- [ ] **Step 4: Commit**

`git add . && git commit -m "chore: scaffold repo, data fetcher, ignores"`

### Task 2: `features.py` — shared data pipeline

**Files:**
- Create: `features.py`
- Test: `test_features.py`

**Interfaces:**
- Produces (relied on by Tasks 3–5):
  - `TARGET = "fraud_bool"`, `SENTINEL_COLS: list[str]`,
    `CATEGORICAL: list[str]`, `NUMERIC: list[str]` (computed as all non-target,
    non-categorical cols)
  - `load_df(path="data/Base.csv") -> pd.DataFrame`
  - `clean(df) -> pd.DataFrame` — SENTINEL_COLS: `-1` → NaN, add
    `<col>_was_missing` int8 indicator; numeric NaN → median; returns new df
  - `make_splits(df) -> (train, val, test)` — 60/20/20 stratified on TARGET,
    seed 42
  - `to_dicts(df) -> list[dict]` — records for DictVectorizer
- Consumes: nothing.

- [ ] **Step 1: Write failing tests**

```python
def test_sentinel_becomes_indicator_and_fill():
    df = pd.DataFrame({"credit_risk_score": [-1, 100, 200],
                       "payment_type": ["A", "B", "A"],
                       "fraud_bool": [0, 1, 0]})
    out = clean(df)
    assert "credit_risk_score_was_missing" in out.columns
    assert out.loc[0, "credit_risk_score_was_missing"] == 1
    assert out["credit_risk_score"].notna().all()
    assert out.loc[0, "credit_risk_score"] == 150  # median fill

def test_splits_are_stratified_60_20_20():
    df = load_df()  # real file; slow-ish, acceptable
    tr, va, te = make_splits(df)
    assert len(tr) == 600_000 and len(va) == 200_000 and len(te) == 200_000
    rates = [s.fraud_bool.mean() for s in (tr, va, te)]
    assert max(rates) - min(rates) < 0.001
```

- [ ] **Step 2: Run tests → verify FAIL** (`pytest test_features.py -v`)

- [ ] **Step 3: Implement `features.py`**

`load_df` = `pd.read_csv`. `clean`: sentinel→NaN + indicators + median fill
(median computed on the df passed in — caller passes train split first, and
`predict.py` refits nothing: it applies stored medians — see Task 5 note).
`make_splits`: two `train_test_split` calls (80/20 then 75/25 → 60/20/20),
`stratify=df[TARGET]`, `random_state=42`. `to_dicts` = `df.to_dict("records")`.

- [ ] **Step 4: Run tests → verify PASS**

- [ ] **Step 5: Commit** `feat: shared data pipeline (clean, split, dicts)`

### Task 3: `notebook.ipynb` — EDA, tuning, selection

**Files:**
- Create: `notebook.ipynb` (built via cells, executed with
  `jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`)

**Interfaces:**
- Consumes: `features.load_df`, `features.clean`, `features.make_splits`,
  `features.NUMERIC`, `features.CATEGORICAL`, `features.TARGET`.
- Produces: `CHOSEN_MODEL` dict at top of Task 4 — {name, params} read off the
  notebook's comparison table.

- [ ] **Step 1: Notebook skeleton + data load**

Cells: title/intro md (problem, dataset, license, leakage note:
`device_fraud_count` examined; `month` kept — used for temporal check),
imports, load via `features`, shape/dtypes/head.

- [ ] **Step 2: EDA cells**

Target distribution; missing-value map (sentinel counts); distributions of
key numerics (velocity_*, credit_risk_score, session_length); categoricals
counts; fraud-rate-by-month; correlation matrix on numerics; mutual
information `sklearn.feature_selection.mutual_info_classif` on subsample —
top-15 bar chart. Each cell preceded by a short md cell. `assert df.shape ==
(1_000_000, 32)` as a guard cell.

- [ ] **Step 3: Feature importance + leakage audit**

Permutation importance of a quick fitted RF on the 200k subsample (top-15
chart); explicit check that `device_fraud_count` importance is noted as
near-leakage (comment on whether to keep — keep, it's real signal, flag in
md).

- [ ] **Step 4: Baseline + tuned models on subsample**

200k stratified subsample → same 60/20/20 split via `make_splits`.
DictVectorizer fit on train only. Models: `DummyClassifier` (PR-AUC floor),
`LogisticRegression` tune `C ∈ {0.01, 0.1, 1, 10}`, `RandomForest` tune
`max_depth ∈ {8, 16} × n_estimators ∈ {100, 200}` class_weight='balanced',
`XGBClassifier` tune `max_depth ∈ {4, 6} × learning_rate ∈ {0.05, 0.1}`
scale_pos_weight=neg/pos. Score each on val: ROC-AUC, PR-AUC, recall@cap-5%.
Markdown summary table; pick winner by val PR-AUC (tie-break recall@cap).

- [ ] **Step 5: Threshold analysis + temporal holdout**

PR curve for winner on val; threshold sweep → table of flag-rate vs
recall/precision; select threshold = max recall s.t. flag-rate ≤5% (also
report F1-optimal). Temporal eval: refit winner on months ≤5 of the FULL
train, test on months 6–7 — report ROC-AUC/PR-AUC vs random-split numbers
(one honest comparison paragraph).

- [ ] **Step 6: Execute notebook end-to-end + commit**

`jupyter nbconvert --to notebook --execute --inplace notebook.ipynb` —
must complete without errors. `git add notebook.ipynb && git commit -m
"feat: EDA, feature importance, tuned model comparison, threshold analysis"`

### Task 4: `train.py` → `model.bin`

**Files:**
- Create: `train.py`

**Interfaces:**
- Consumes: `features.*`; `CHOSEN_MODEL` (from Task 3 Step 4's table).
- Produces: `model.bin` = pickled
  `{"model": fitted_model, "dv": fitted_DictVectorizer,
    "threshold": float, "features": list[str],
    "medians": dict[str, float], "metrics": dict}`.
  CLI: `python train.py` → prints test metrics + artifact size.

- [ ] **Step 1: Write the artifact contract check**

`test_train.py::test_artifact_contract` — run `train.py` with env var
`TRAIN_SAMPLE=50000` (fast mode: subsamples train for tests/dev), then
`pickle.load("model.bin")` and assert keys
`{"model","dv","threshold","features","medians","metrics"}` present,
`0 < threshold < 1`, and file size < 100 MB.

- [ ] **Step 2: Run → FAIL** (no train.py yet)

- [ ] **Step 3: Implement `train.py`**

Flow: load → clean (medians computed on train split only, stored) → splits
→ fit model A = CHOSEN_MODEL params on train → threshold from val at ≤5%
flag → refit same config on train+val → evaluate test → pack dict → pickle.
`TRAIN_SAMPLE` env subsamples the train pool for dev runs. Print metrics.

- [ ] **Step 4: Run tests → PASS; run full `python train.py`**

Full run on 1M rows (XGBoost ~minutes). Record printed metrics — they go in
the README table.

- [ ] **Step 5: Commit** `feat: train script producing model.bin`

### Task 5: `predict.py` + `test_predict.py`

**Files:**
- Create: `predict.py`, `test_predict.py`

**Interfaces:**
- Consumes: `model.bin` keys from Task 4; `features.clean`, `features.to_dicts`
  (clean must accept pre-computed medians → see Task 4 note: add optional
  `medians=None` param; when given, fill uses them instead of computing).
- Produces: `POST /predict` → `{"fraud_score": float, "decision": bool}`;
  `GET /health` → `{"status": "ok"}`.

- [ ] **Step 1: Write the API test**

`test_predict.py` with `TestClient(app)` cases: health 200; valid full
request → 200 + both fields; missing field → 422; `-1` sentinel fields → 200
same-shape response; unknown `device_os="haiku_os"` → 200; extra field → 200.
One representative valid payload literal in the file.

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement `predict.py`**

FastAPI app; startup loads `model.bin` — if missing, `RuntimeError` naming
`python train.py`. Pydantic model = all 30 feature fields as
`Optional[float|str]` with a root validator that 422s listing absent fields;
`model_config = ConfigDict(extra="ignore")`. Request → `clean` (stored
medians) → `to_dicts` → `dv.transform` → `predict_proba` → response.

- [ ] **Step 4: Run tests → PASS; live check**

`uvicorn predict:app --port 9696` + `python test_predict.py --live` prints
score for a real fraud row and a real legit row from test split.

- [ ] **Step 5: Commit** `feat: FastAPI scoring service`

### Task 6: requirements.txt, Dockerfile, container verify

**Files:**
- Modify: `requirements.txt` (pinned via `pip freeze` of project deps)
- Create: `Dockerfile`

- [ ] **Step 1: Pin deps**

`pip freeze` the .venv → filter to project imports → `requirements.txt`.

- [ ] **Step 2: Write Dockerfile**

`FROM python:3.12-slim` (xgboost wheels safe); `WORKDIR /app`;
`COPY requirements.txt` → `pip install --no-cache-dir -r`;
`COPY predict.py features.py model.bin .`; `EXPOSE 9696`;
`CMD ["uvicorn","predict:app","--host","0.0.0.0","--port","9696"]`.
(model.bin is gitignored but copied at build time — Docker build runs after
`train.py` produces it. README documents this order.)

- [ ] **Step 3: Build + run + verify**

`docker build -t bank-fraud .` then
`docker run --rm -p 9696:9696 bank-fraud` → `curl localhost:9696/health` 200;
POST a sample body → score + decision. Record the exact commands for README.

- [ ] **Step 4: Commit** `feat: pinned deps, Dockerfile, verified container`

### Task 7: README, cloud attempt, fresh-clone check, submission prep

**Files:**
- Create: `README.md`, `screenshots/` (evidence)
- Modify: nothing else

- [ ] **Step 1: README**

Sections (project-tips order): problem + how the model is used (threshold
policy as business story) · dataset (source, license CC-BY-NC-SA,
semi-synthetic disclosure, `get_data.py` instructions incl. manual fallback)
· EDA summary w/ key findings · model comparison table + chosen model +
metrics (test ROC-AUC/PR-AUC/recall@cap, temporal-check numbers) · run
locally (venv) · run Docker (exact commands from Task 6) · API example
request/response · limitations + next steps (calibration, drift monitoring).
Screenshot(s) of working service in `screenshots/`.

- [ ] **Step 2: Cloud deploy attempt (user-dependent)**

Needs a GitHub repo under the user's account + a Render/Railway/HF account —
**stop and ask user**; push `git remote` only with their repo URL/consent.
Fallback: screenshot/video evidence satisfies deployment criterion.

- [ ] **Step 3: Fresh-clone verification**

`git clone` to temp dir → follow README verbatim (minus 1M-row retrain: run
`TRAIN_SAMPLE` dev mode + docker build) → confirm no missing steps; fix
README gaps found.

- [ ] **Step 4: Final commit + note hash**

`git log -1 --format=%H` → record for submission form.

---

## Self-Review Notes

- Spec coverage: all rubric rows map to tasks — EDA (T3), tuning (T3),
  train script (T4), service (T5), deps+Docker (T6), README+cloud (T7).
- `clean()` signature gained `medians=None` — consumed in T5, stored in T4's
  artifact dict; single-signature change noted in both tasks.
- Notebook is not TDD-able; its verification = clean `nbconvert` execution +
  `assert` guard cells.
