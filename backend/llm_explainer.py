"""
LLM Explainer Layer + Dynamic Enrichment
Handles fast, resilient merchant explanations, correct risk scoring thresholds,
and WhatsApp action links.
"""

import os
import re
import json
import urllib.parse
from google import genai
from google.genai import types

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

SUSPICIOUS_TERMS = [
    "review", "credit score", "bank statement", "income",
    "employment", "identity", "police", "criminal", "verified", 
    "approved", "rejected",
]


def add_interaction_features(order: dict) -> dict:
    """
    Engineers interaction features required by the prediction model.
    """
    order["high_value_cod"] = int(
        (order.get("order_value", 0) > 3000) and (order.get("is_prepaid") == 0)
    )
    order["risky_address_cod"] = int(
        (order.get("address_completeness", 100) < 70) and (order.get("is_prepaid") == 0)
    )
    return order


def get_risk_tier(risk_score: float) -> str:
    """
    Risk Tier Thresholds:
    - LOW: < 0.50
    - BORDERLINE: 0.50 to 0.75
    - HIGH: > 0.75
    """
    if risk_score < 0.50:
        return "LOW"
    elif risk_score <= 0.75:
        return "BORDERLINE"
    else:
        return "HIGH"


def generate_whatsapp_link(phone_number: str, message: str) -> str:
    """
    Generates a deep-link URL that opens WhatsApp with a pre-filled custom message.
    """
    if not message:
        return ""
        
    clean_phone = re.sub(r"[^\d]", "", str(phone_number or ""))
    encoded_message = urllib.parse.quote(message)
    if clean_phone:
        return f"https://wa.me/{clean_phone}?text={encoded_message}"
    return f"https://wa.me/?text={encoded_message}"


def detect_primary_issue(top_features: list, order: dict, risk_tier: str) -> dict:
    """Analyzes top features and risk tier to identify actionable issue."""
    
    # LOW risk orders require no verification or actions
    if risk_tier == "LOW":
        return {
            "issue_type": "none",
            "description": "Order parameters match expected safe order profile",
            "action_needed": "Approve for immediate shipping."
        }

    top_feature_names = [name for name, _ in top_features[:3]]

    if "address_completeness" in top_feature_names and order.get("address_completeness", 100) < 70:
        return {
            "issue_type": "incomplete_address",
            "description": "Address details incomplete (missing house number, PIN, or landmark)",
            "action_needed": "Please share your complete address including house number and PIN code."
        }

    if "order_value" in top_feature_names and order.get("order_value", 0) > 3000 and order.get("is_prepaid") == 0:
        return {
            "issue_type": "high_value_cod",
            "description": "High-value COD order — needs confirmation from customer",
            "action_needed": "Please confirm you want to place this cash-on-delivery order."
        }

    if "past_cod_rejections" in top_feature_names and order.get("past_cod_rejections", 0) >= 2:
        return {
            "issue_type": "repeat_rejections",
            "description": f"Customer has {order.get('past_cod_rejections')} past rejection(s) — extra verification needed",
            "action_needed": "Verify delivery address and availability before shipping."
        }

    return {
        "issue_type": "generic_verify",
        "description": "Order needs standard verification before shipping",
        "action_needed": "Please confirm your order details (address and phone number)."
    }


def explain_order(order: dict, risk_score: float, top_features: list) -> dict:
    risk_tier = get_risk_tier(risk_score)
    issue = detect_primary_issue(top_features, order, risk_tier)

    # 1. Clear logic path for LOW Risk Orders
    if risk_tier == "LOW":
        explanation = f"Order risk is LOW ({int(risk_score * 100)}%). Standard safe order metrics met."
        whatsapp_msg = ""
        suggested_action = "Approve for immediate shipping."
        needs_human_review = False
        review_reason = None
        wa_link = ""

    # 2. Logic path for BORDERLINE and HIGH Risk Orders
    else:
        explanation = f"Order risk is {risk_tier} ({int(risk_score * 100)}%). Primary factor: {issue['description']}."
        whatsapp_msg = f"Hi! Please confirm your order details so we can process your order promptly."
        suggested_action = issue["action_needed"]
        
        # Both BORDERLINE and HIGH tiers require human inspection/confirmation
        needs_human_review = risk_tier in ["BORDERLINE", "HIGH"]
        review_reason = f"Model risk score falls in {risk_tier} zone ({int(risk_score * 100)}%)" if needs_human_review else None

        if client:
            prompt = f"""You are a merchant order risk explainer.
Risk Score Tier: {risk_tier} ({risk_score:.2f})
Top Risk Features: {top_features[:3]}
Primary Issue: {issue['description']}
Action Needed: {issue['action_needed']}

Return a valid JSON object with:
1. "explanation": 1-2 sentence merchant risk explanation stating that risk tier is {risk_tier}.
2. "whatsapp_message": Friendly customer verification message asking customer to confirm order.
"""
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=300,
                        response_mime_type="application/json",
                    ),
                )
                data = json.loads(response.text)
                explanation = data.get("explanation", explanation)
                whatsapp_msg = data.get("whatsapp_message", whatsapp_msg)
            except Exception:
                explanation += " (AI enrichment temporarily using offline mode)."

        wa_link = generate_whatsapp_link(order.get("phone_number", ""), whatsapp_msg)

    validate_llm_output(explanation)
    validate_numeric_claims(explanation, order)

    return {
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "explanation": explanation,
        "suggested_action": suggested_action,
        "needs_human_review": needs_human_review,
        "review_reason": review_reason,
        "issue_type": issue["issue_type"],
        "issue_description": issue["description"],
        "whatsapp_message": whatsapp_msg,
        "whatsapp_link": wa_link,
    }


def validate_llm_output(llm_text: str) -> dict:
    text_lower = llm_text.lower()
    found_suspicious = [t for t in SUSPICIOUS_TERMS if t in text_lower]
    return {
        "has_hallucination_signal": len(found_suspicious) > 0,
        "suspicious_terms_found": found_suspicious
    }


def validate_numeric_claims(llm_text: str, order_features: dict) -> dict:
    text_lower = llm_text.lower()
    mismatches = []

    m = re.search(r"(\d+)\s*(?:past\s+)?(?:cod\s+)?rejections?", text_lower)
    if m and "past_cod_rejections" in order_features:
        claimed = int(m.group(1))
        actual = int(order_features["past_cod_rejections"])
        if claimed != actual:
            mismatches.append(f"claimed {claimed} past rejections, actual is {actual}")

    return {
        "has_numeric_mismatch": len(mismatches) > 0,
        "mismatches_found": mismatches
    }


def get_top_features_for_order(model, order: dict, feature_names: list) -> list:
    """Gets per-order feature contributions safely across both LGBMClassifier and Booster."""
    import pandas as pd

    row_df = pd.DataFrame([order])[feature_names].copy()
    if "item_category" in row_df.columns:
        row_df["item_category"] = row_df["item_category"].astype("category")

    booster = model.booster_ if hasattr(model, "booster_") else model

    try:
        contributions = booster.predict(row_df, pred_contrib=True)[0]
        if len(contributions) == len(feature_names) + 1:
            contributions = contributions[:-1]
        feature_contrib_pairs = list(zip(feature_names, contributions))
        top_features = sorted(feature_contrib_pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
    except Exception:
        top_features = [(name, 0.0) for name in feature_names[:3]]

    return [(name, order.get(name, 0)) for name, _ in top_features]