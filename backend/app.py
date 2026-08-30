"""
FastAPI Backend — Return Risk Scorer
======================================
WHAT THIS DOES:
Exposes one main endpoint, POST /predict, that a merchant's checkout
system would call the moment an order is placed. It:
  1. Takes raw order details (payment mode, city tier, order value, etc.)
  2. Computes the same engineered features used in training
  3. Runs the LightGBM model to get a risk_score
  4. Passes that to the LLM explainer for a plain-English explanation
     + suggested action + human-review flag
  5. Returns everything as one JSON response

WHY FASTAPI:
- Automatic request validation (Pydantic) — bad input gets rejected
  with a clear error before it ever reaches our model
- Auto-generated docs at /docs — useful for your demo video, judges
  can literally test your API in the browser without writing code
- Async-ready, fast — matches the "real-time" framing of the problem
"""

import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Let this file import from backend/llm_explainer.py regardless of
# which folder uvicorn is launched from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from llm_explainer import (
    explain_order,
    add_interaction_features,
    get_top_features_for_order,
)

app = FastAPI(
    title="Return Risk Scorer API",
    description="Predicts COD order rejection risk and suggests merchant actions",
    version="1.0.0",
)

# Allows the Streamlit dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = ["is_prepaid", "city_tier", "past_cod_rejections", "order_value",
            "item_category", "address_completeness", "order_hour", "is_weekend",
            "high_value_cod", "risky_address_cod"]

# Load model once at startup, not per-request — loading from disk every
# request would be slow and pointless since the model doesn't change
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lightgbm_model.pkl")
lgb_model = joblib.load(MODEL_PATH)


class OrderRequest(BaseModel):
    """
    Defines exactly what a valid request looks like. FastAPI uses this
    to auto-validate incoming requests — e.g. order_value must be a
    positive number, is_prepaid must be 0 or 1, etc. Bad requests get
    rejected automatically with a clear error, before touching our model.
    """
    order_id: str = Field(..., example="ORD00123")
    is_prepaid: int = Field(..., ge=0, le=1, description="0 = COD, 1 = Prepaid")
    city_tier: int = Field(..., ge=1, le=3)
    past_cod_rejections: int = Field(..., ge=0)
    order_value: float = Field(..., gt=0)
    item_category: str = Field(..., example="fashion")
    address_completeness: float = Field(..., ge=0, le=100)
    order_hour: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    order_id: str
    risk_score: float
    risk_tier: str
    explanation: str
    suggested_action: str
    needs_human_review: bool
    review_reason: str | None
    top_features: list[list]  # [[feature_name, value], ...] — top 3 drivers of this order's score


@app.get("/")
def root():
    return {"status": "Return Risk Scorer API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": lgb_model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(order: OrderRequest):
    try:
        order_dict = order.model_dump(exclude={"order_id"})
        order_dict = add_interaction_features(order_dict)

        row = pd.DataFrame([order_dict])[FEATURES]
        row["item_category"] = row["item_category"].astype("category")

        risk_score = float(lgb_model.predict_proba(row)[0][1])
        top_features = get_top_features_for_order(lgb_model, order_dict, FEATURES)

        result = explain_order(order_dict, risk_score, top_features)
        result["order_id"] = order.order_id
        result["top_features"] = [[name, value] for name, value in top_features]
        return result

    except Exception as e:
        # Surface a clean error instead of a raw stack trace to whoever
        # is calling this API (dashboard, judges testing via /docs, etc.)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)