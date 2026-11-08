import os
import subprocess
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import features
from predict import app

client = TestClient(app)

# one representative valid application (a real legit row's values)
VALID = {
    "income": 0.7, "name_email_similarity": 0.99,
    "prev_address_months_count": 12, "current_address_months_count": 120,
    "customer_age": 40, "days_since_request": 0.0,
    "intended_balcon_amount": 15.0, "payment_type": "AB",
    "zip_count_4w": 3000, "velocity_6h": 2500.0, "velocity_24h": 6000.0,
    "velocity_4w": 8000.0, "bank_branch_count_8w": 3,
    "date_of_birth_distinct_emails_4w": 6, "employment_status": "CA",
    "credit_risk_score": 200, "email_is_free": 1, "housing_status": "BC",
    "phone_home_valid": 1, "phone_mobile_valid": 1, "bank_months_count": 30,
    "has_other_cards": 1, "proposed_credit_limit": 500.0,
    "foreign_request": 0, "source": "INTERNET",
    "session_length_in_minutes": 10.0, "device_os": "windows",
    "keep_alive_session": 1, "device_distinct_emails_8w": 1,
    "device_fraud_count": 0, "month": 3,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_valid_request_scores():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"fraud_score", "decision"}
    assert isinstance(body["fraud_score"], float)
    assert isinstance(body["decision"], bool)


def test_missing_field_422():
    bad = {k: v for k, v in VALID.items() if k != "credit_risk_score"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert "credit_risk_score" in r.text


def test_sentinel_minus_one_handled():
    req = {**VALID, "prev_address_months_count": -1,
           "session_length_in_minutes": -1}
    r = client.post("/predict", json=req)
    assert r.status_code == 200
    assert set(r.json()) == {"fraud_score", "decision"}


def test_unknown_category_still_scores():
    r = client.post("/predict", json={**VALID, "device_os": "haiku_os"})
    assert r.status_code == 200


def test_extra_fields_ignored():
    r = client.post("/predict", json={**VALID, "hacker_field": "x"})
    assert r.status_code == 200


def test_missing_model_names_fix(tmp_path):
    env = {**os.environ, "MODEL_PATH": str(tmp_path / "nope.bin")}
    r = subprocess.run([sys.executable, "-c", "import predict"],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "python train.py" in r.stderr


if __name__ == "__main__" and "--live" in sys.argv:
    import requests

    df = features.load_df()
    tr, va, te = features.make_splits(df)
    for label, y in (("legit", 0), ("fraud", 1)):
        row = te[te.fraud_bool == y].drop(columns=[features.TARGET]).iloc[0]
        r = requests.post("http://localhost:9696/predict", json=row.to_dict())
        print(f"{label}: {r.json()}")
