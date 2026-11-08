"""Train the final fraud model and save model.bin.

No hyperparameter search here — CHOSEN_MODEL is the config the notebook's
comparison selected (XGBoost d=4, lr=0.05). Flow: split -> medians on train
-> clean -> fit on train -> capacity threshold on val -> refit on train+val
-> evaluate on test -> pickle artifact.

Env:
  TRAIN_SAMPLE  subsample the dataset before splitting (dev/test speed)
  OUTPUT        artifact path (default model.bin)
"""
import os
import pickle
import sys

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import features

CHOSEN_MODEL = {
    "name": "xgb",
    "params": {
        "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
}
CAP = 0.05  # review capacity: flag-rate ceiling for threshold selection


def make_model(y):
    spw = float((y == 0).sum() / max(1, (y == 1).sum()))
    return XGBClassifier(**CHOSEN_MODEL["params"], scale_pos_weight=spw,
                         eval_metric="aucpr", random_state=features.SEED,
                         n_jobs=-1)


def flag_stats(y, s, threshold):
    flag = s >= threshold
    return dict(flag_rate=float(flag.mean()),
                recall=float(y[flag].sum() / max(1, y.sum())),
                precision=float(y[flag].sum() / max(1, flag.sum())))


def main():
    n = int(os.environ.get("TRAIN_SAMPLE", "0"))
    out_path = os.environ.get("OUTPUT", "model.bin")

    df = features.load_df()
    if n:
        df = df.sample(n=min(n, len(df)), random_state=features.SEED)

    tr, va, te = features.make_splits(df)
    med = features.medians_of(tr)
    tr, va, te = (features.clean(d, med) for d in (tr, va, te))

    ytr, yva, yte = (d[features.TARGET].values for d in (tr, va, te))
    feat_cols = [c for c in tr.columns if c != features.TARGET]

    # 1) fit on train -> threshold on val (flag-rate <= CAP)
    dv = DictVectorizer(sparse=False)
    Xtr = dv.fit_transform(features.to_dicts(tr[feat_cols]))
    Xva = dv.transform(features.to_dicts(va[feat_cols]))

    m_a = make_model(ytr).fit(Xtr, ytr)
    sva = m_a.predict_proba(Xva)[:, 1]
    threshold = float(np.quantile(sva, 1 - CAP))

    # 2) refit same config on train+val, evaluate on test
    trv = np.concatenate([Xtr, Xva])
    ytrv = np.concatenate([ytr, yva])
    model = make_model(ytrv).fit(trv, ytrv)

    Xte = dv.transform(features.to_dicts(te[feat_cols]))
    ste = model.predict_proba(Xte)[:, 1]
    metrics = dict(roc_auc=float(roc_auc_score(yte, ste)),
                   pr_auc=float(average_precision_score(yte, ste)),
                   threshold=threshold,
                   **{f"at_cap_{k}": v
                      for k, v in flag_stats(yte, ste, threshold).items()})

    art = {"model": model, "dv": dv, "threshold": threshold,
           "features": dv.get_feature_names_out().tolist(),
           "medians": med, "metrics": metrics}
    with open(out_path, "wb") as f:
        pickle.dump(art, f)

    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"model: {CHOSEN_MODEL['name']} {CHOSEN_MODEL['params']}")
    print(f"test metrics: {metrics}")
    print(f"artifact: {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
