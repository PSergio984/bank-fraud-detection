"""FastAPI service: score bank account applications for fraud.

POST /predict  -> {"fraud_score": float, "decision": bool}
GET  /health   -> {"status": "ok"}

fraud_score is the model's rank-consistent score (not a calibrated
probability — class weighting shifts outputs). decision = score >= the
review-capacity threshold chosen on validation (<=5% flag rate).
"""
import os
import pickle
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, model_validator

import features

MODEL_PATH = os.environ.get("MODEL_PATH", "model.bin")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"{MODEL_PATH} not found — train the model first: python train.py")
with open(MODEL_PATH, "rb") as f:
    ART = pickle.load(f)

app = FastAPI(title="bank-fraud-detector")


class Application(BaseModel):
    model_config = ConfigDict(extra="ignore")

    income: Optional[float] = None
    name_email_similarity: Optional[float] = None
    prev_address_months_count: Optional[float] = None
    current_address_months_count: Optional[float] = None
    customer_age: Optional[float] = None
    days_since_request: Optional[float] = None
    intended_balcon_amount: Optional[float] = None
    payment_type: Optional[str] = None
    zip_count_4w: Optional[float] = None
    velocity_6h: Optional[float] = None
    velocity_24h: Optional[float] = None
    velocity_4w: Optional[float] = None
    bank_branch_count_8w: Optional[float] = None
    date_of_birth_distinct_emails_4w: Optional[float] = None
    employment_status: Optional[str] = None
    credit_risk_score: Optional[float] = None
    email_is_free: Optional[float] = None
    housing_status: Optional[str] = None
    phone_home_valid: Optional[float] = None
    phone_mobile_valid: Optional[float] = None
    bank_months_count: Optional[float] = None
    has_other_cards: Optional[float] = None
    proposed_credit_limit: Optional[float] = None
    foreign_request: Optional[float] = None
    source: Optional[str] = None
    session_length_in_minutes: Optional[float] = None
    device_os: Optional[str] = None
    keep_alive_session: Optional[float] = None
    device_distinct_emails_8w: Optional[float] = None
    device_fraud_count: Optional[float] = None
    month: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def require_all_fields(cls, data):
        missing = [f for f in cls.model_fields
                   if not isinstance(data, dict) or data.get(f) is None]
        if missing:
            raise ValueError(f"missing required fields: {missing}")
        return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Application):
    df = features.clean(pd.DataFrame([req.model_dump()]), ART["medians"])
    X = ART["dv"].transform(features.to_dicts(df))
    score = float(ART["model"].predict_proba(X)[0, 1])
    return {"fraud_score": score, "decision": score >= ART["threshold"]}
