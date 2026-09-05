"""
FastAPI Backend — Return Risk Scorer
"""

import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from llm_explainer import (
    explain_order,
    add_interaction_features,
    get_top_features_for_order,
)
from address_scorer import score_address_completeness

app = FastAPI(
    title="Return Risk Scorer API",
    description="Predicts COD order rejection risk and suggests merchant actions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES = [
    "is_prepaid", "city_tier", "past_cod_rejections", "order_value",
    "item_category", "address_completeness", "order_hour", "is_weekend",
    "high_value_cod", "risky_address_cod"
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lightgbm_model.pkl")

try:
    lgb_model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model at {MODEL_PATH}: {e}")
    lgb_model = None


class OrderRequest(BaseModel):
    order_id: str = Field(..., example="ORD00123")
    is_prepaid: int = Field(..., ge=0, le=1, description="0 = COD, 1 = Prepaid")
    city_tier: int = Field(..., ge=1, le=3)
    past_cod_rejections: int = Field(..., ge=0)
    order_value: float = Field(..., gt=0)
    item_category: str = Field(..., example="fashion")
    address_text: str = Field(..., example="H.No. 8, MG Road, Near City Hospital, Mumbai, 400001")
    phone_number: str = Field(default="", example="919876543210")
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
    top_features: list[list]
    address_completeness: int
    issue_type: str
    issue_description: str
    whatsapp_message: str
    whatsapp_link: str


@app.get("/")
def root():
    return {"status": "Return Risk Scorer API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": lgb_model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(order: OrderRequest):
    try:
        order_dict = order.model_dump(exclude={"order_id", "address_text"})
        order_dict["address_completeness"] = score_address_completeness(order.address_text)
        order_dict = add_interaction_features(order_dict)

        if lgb_model is not None:
            row = pd.DataFrame([order_dict])[FEATURES]
            row["item_category"] = row["item_category"].astype("category")

            risk_score = float(lgb_model.predict_proba(row)[0][1])
            top_features = get_top_features_for_order(lgb_model, order_dict, FEATURES)
        else:
            risk_score = 0.25
            top_features = [
                ("is_prepaid", order_dict["is_prepaid"]),
                ("past_cod_rejections", order_dict["past_cod_rejections"]),
                ("order_value", order_dict["order_value"]),
            ]

        result = explain_order(order_dict, risk_score, top_features)

        # Enforce Low Risk Reset
        if result.get("risk_tier") == "LOW" or risk_score < 0.50:
            result["risk_tier"] = "LOW"
            result["explanation"] = f"Order risk is LOW ({risk_score:.0%}). Standard safe order metrics met."
            result["suggested_action"] = "Approve for immediate shipping."
            result["needs_human_review"] = False
            result["review_reason"] = None
            result["issue_type"] = "none"
            result["issue_description"] = "No verification needed."
            result["whatsapp_message"] = ""
            result["whatsapp_link"] = ""

        result["order_id"] = order.order_id
        result["risk_score"] = risk_score
        result["top_features"] = [[name, value] for name, value in top_features]
        result["address_completeness"] = order_dict["address_completeness"]

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)