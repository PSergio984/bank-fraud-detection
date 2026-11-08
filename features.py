"""Shared data pipeline for the BAF fraud-detection project.

Used identically by notebook.ipynb, train.py and predict.py so training and
serving see the same cleaning, splits and feature lists.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "fraud_bool"
SEED = 42

# -1 is the dataset's missing-value sentinel in these columns
SENTINEL_COLS = [
    "prev_address_months_count",
    "current_address_months_count",
    "credit_risk_score",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

CATEGORICAL = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

NUMERIC = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "email_is_free", "phone_home_valid",
    "phone_mobile_valid", "bank_months_count", "has_other_cards",
    "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session",
    "device_distinct_emails_8w", "device_fraud_count", "month",
]


def load_df(path="data/Base.csv"):
    return pd.read_csv(path)


def medians_of(df):
    """Median per numeric column, sentinel values excluded — computed on the
    train split, stored in the artifact so serving fills identically."""
    out = {}
    for c in NUMERIC:
        if c in df.columns:
            s = df[c].replace(-1, pd.NA) if c in SENTINEL_COLS else df[c]
            out[c] = float(pd.to_numeric(s, errors="coerce").median())
    return out


def clean(df, medians=None):
    """-1 sentinels -> NaN (+ *_was_missing indicator), numeric NaN -> median.

    medians=None computes them from df (training time); pass the stored
    dict at serving time so nothing refits."""
    df = df.copy()
    if medians is None:
        medians = medians_of(df)
    for col in SENTINEL_COLS:
        if col not in df.columns:
            continue
        df[col + "_was_missing"] = (df[col] == -1).astype("int8")
        df[col] = df[col].replace(-1, pd.NA)
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(medians[col])
    return df


def make_splits(df, seed=SEED):
    """60/20/20 stratified on TARGET -> (train, val, test)."""
    train, rest = train_test_split(
        df, test_size=0.4, random_state=seed, stratify=df[TARGET])
    val, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=rest[TARGET])
    return train, val, test


def to_dicts(df):
    return df.to_dict("records")
