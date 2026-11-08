import pandas as pd

from features import clean, load_df, make_splits


def test_sentinel_becomes_indicator_and_fill():
    df = pd.DataFrame({"credit_risk_score": [-1, 100, 200],
                       "payment_type": ["A", "B", "A"],
                       "fraud_bool": [0, 1, 0]})
    out = clean(df)
    assert "credit_risk_score_was_missing" in out.columns
    assert out.loc[0, "credit_risk_score_was_missing"] == 1
    assert out["credit_risk_score"].notna().all()
    assert out.loc[0, "credit_risk_score"] == 150  # median fill


def test_clean_uses_stored_medians_when_given():
    df = pd.DataFrame({"credit_risk_score": [-1, 300],
                       "payment_type": ["A", "B"],
                       "fraud_bool": [0, 1]})
    out = clean(df, medians={"credit_risk_score": 500})
    assert out.loc[0, "credit_risk_score"] == 500


def test_splits_are_stratified_60_20_20():
    df = load_df()  # real file; slow-ish, acceptable
    tr, va, te = make_splits(df)
    assert len(tr) == 600_000 and len(va) == 200_000 and len(te) == 200_000
    rates = [s.fraud_bool.mean() for s in (tr, va, te)]
    assert max(rates) - min(rates) < 0.001
