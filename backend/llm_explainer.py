import os
import json
import warnings
from dotenv import load_dotenv
from google import genai
from google.genai import types

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.6-flash"  # gemini-2.5-flash and gemini-1.5-flash are both retired

LOW_RISK_THRESHOLD = 0.40
HIGH_RISK_THRESHOLD = 0.65

# Features the LLM is allowed to reference — validator uses this to catch
# hallucinations (e.g. LLM inventing "bad reviews" which isn't real data)
SUSPICIOUS_TERMS = ["review", "social media", "called the customer", "email bounced",
                     "credit score", "bank statement", "previous complaint"]


def get_risk_tier(risk_score: float) -> str:
    if risk_score < LOW_RISK_THRESHOLD:
        return "LOW"
    elif risk_score <= HIGH_RISK_THRESHOLD:
        return "BORDERLINE"
    return "HIGH"


def build_prompt(risk_score: float, top_features: list, risk_tier: str) -> str:
    features_text = "\n".join([f"- {name}: {value}" for name, value in top_features])

    if risk_tier == "LOW":
        allowed_actions = "[ship normally]"
        tone_instruction = "Explain briefly why the order appears safe and trustworthy."
    elif risk_tier == "BORDERLINE":
        allowed_actions = "[verify address via WhatsApp, request partial deposit, flag for manual check]"
        tone_instruction = "Explain why the risk is uncertain and recommend a lightweight check."
    else:
        allowed_actions = "[request prepaid, request partial deposit, verify address via WhatsApp]"
        tone_instruction = "Explain the clear red flags driving the high risk score."

    return f"""You are a risk explanation assistant for an Indian e-commerce merchant.
An ML model calculated this order's rejection risk score as {risk_score:.0%} (Tier: {risk_tier}).
Top feature drivers (ONLY use these, do not invent others):
{features_text}
Rules:
- {tone_instruction}
- Provide 1 short sentence for 'explanation' and 1 clear action for 'suggested_action'.
- Choose 'suggested_action' ONLY from: {allowed_actions}.
- Write simply for a merchant. Do not invent unlisted details.
Respond ONLY in JSON: {{"explanation": "...", "suggested_action": "..."}}"""


def validate_llm_output(llm_text: str) -> dict:
    """
    Checks the LLM's explanation for terms that suggest it invented data
    we never gave it. This is our hallucination safety net — independent
    of the risk_tier, so even a HIGH or LOW confidence order gets checked.
    """
    text_lower = llm_text.lower()
    found_suspicious = [t for t in SUSPICIOUS_TERMS if t in text_lower]
    return {
        "has_hallucination_signal": len(found_suspicious) > 0,
        "suspicious_terms_found": found_suspicious
    }


def add_interaction_features(order_features: dict) -> dict:
    """
    Recreates the same 2 engineered features used during training
    (models/train.py) — must match exactly, or the model's expected
    input shape (10 features) won't line up with what we send it.
    """
    order_features = dict(order_features)  # don't mutate the original
    order_features["high_value_cod"] = int(
        order_features["order_value"] > 3000 and order_features["is_prepaid"] == 0
    )
    order_features["risky_address_cod"] = int(
        order_features["address_completeness"] < 50 and order_features["is_prepaid"] == 0
    )
    return order_features


def get_top_features_for_order(model, order_features: dict, feature_names: list, top_n: int = 3) -> list:
    """
    Returns the top_n features driving THIS specific order's prediction
    (via LightGBM's per-sample contributions), not the same global
    importance list for every order — makes each explanation order-specific.
    """
    import pandas as pd
    row = pd.DataFrame([order_features])[feature_names]
    if "item_category" in row.columns:
        row["item_category"] = row["item_category"].astype("category")

    contributions = model.predict(row, pred_contrib=True)[0]
    contrib_pairs = list(zip(feature_names, contributions[:-1]))
    contrib_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    return [(name, order_features[name]) for name, _ in contrib_pairs[:top_n]]


def explain_order(order: dict, risk_score: float, top_features: list) -> dict:
    risk_tier = get_risk_tier(risk_score)
    prompt = build_prompt(risk_score, top_features, risk_tier)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2
    )

    fallback_action = "ship normally" if risk_tier == "LOW" else "verify address via WhatsApp"
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt, config=config
        )
        parsed = json.loads(response.text)
        explanation = parsed.get("explanation", "")
        suggested_action = parsed.get("suggested_action", fallback_action)
    except Exception as e:
        explanation = f"Order risk processed based on transactional features. (LLM error: {e})"
        suggested_action = fallback_action

    validation = validate_llm_output(explanation)

    # Two independent triggers for human review — either one is enough:
    # 1. Model itself is unsure (BORDERLINE tier)
    # 2. LLM said something that isn't grounded in the real features
    needs_human_review = (risk_tier == "BORDERLINE") or validation["has_hallucination_signal"]

    if risk_tier == "BORDERLINE" and validation["has_hallucination_signal"]:
        review_reason = f"Borderline score AND possible hallucination: {validation['suspicious_terms_found']}"
    elif risk_tier == "BORDERLINE":
        review_reason = "Model risk score falls in borderline uncertainty zone (40%-65%)"
    elif validation["has_hallucination_signal"]:
        review_reason = f"Possible hallucination: {validation['suspicious_terms_found']}"
    else:
        review_reason = None

    return {
        "risk_score": round(risk_score, 3),
        "risk_tier": risk_tier,
        "explanation": explanation,
        "suggested_action": suggested_action,
        "needs_human_review": needs_human_review,
        "review_reason": review_reason
    }


if __name__ == "__main__":
    import joblib
    import pandas as pd

    lgb_model = joblib.load("models/lightgbm_model.pkl")
    FEATURES = ["is_prepaid", "city_tier", "past_cod_rejections", "order_value",
                "item_category", "address_completeness", "order_hour", "is_weekend",
                "high_value_cod", "risky_address_cod"]

    test_cases = {
        "Clear high risk": {
            "is_prepaid": 0, "city_tier": 3, "past_cod_rejections": 3,
            "order_value": 8200, "item_category": "fashion",
            "address_completeness": 35, "order_hour": 23, "is_weekend": 1
        },
        "Clear low risk": {
            "is_prepaid": 1, "city_tier": 1, "past_cod_rejections": 0,
            "order_value": 900, "item_category": "grocery",
            "address_completeness": 95, "order_hour": 14, "is_weekend": 0
        },
        "Borderline case": {
            "is_prepaid": 0, "city_tier": 1, "past_cod_rejections": 1,
            "order_value": 1500, "item_category": "grocery",
            "address_completeness": 78, "order_hour": 15, "is_weekend": 0
        },
    }

    for label, order_features in test_cases.items():
        order_features = add_interaction_features(order_features)
        row = pd.DataFrame([order_features])[FEATURES]
        row["item_category"] = row["item_category"].astype("category")
        risk_score = lgb_model.predict_proba(row)[0][1]

        top_features = get_top_features_for_order(lgb_model, order_features, FEATURES)
        result = explain_order(order_features, risk_score, top_features)
        print(f"\n=== {label} ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # Scan the test set to find real orders that land in the
    # BORDERLINE zone (40%-65%) — more reliable than hand-picking
    # feature values and hoping they land there.
    # ---------------------------------------------------------
    print("\n\n=== Scanning test set for real BORDERLINE examples ===")
    test_df = pd.read_csv("data/test.csv")
    for df_ in [test_df]:
        df_["high_value_cod"] = ((df_["order_value"] > 3000) & (df_["is_prepaid"] == 0)).astype(int)
        df_["risky_address_cod"] = ((df_["address_completeness"] < 50) & (df_["is_prepaid"] == 0)).astype(int)

    test_rows = test_df[FEATURES].copy()
    test_rows["item_category"] = test_rows["item_category"].astype("category")
    test_df["risk_score"] = lgb_model.predict_proba(test_rows)[:, 1]

    borderline_rows = test_df[(test_df["risk_score"] >= 0.40) & (test_df["risk_score"] <= 0.65)]
    print(f"Found {len(borderline_rows)} real borderline orders in test set")

    if len(borderline_rows) > 0:
        sample = borderline_rows.iloc[0]
        order_features = {f: sample[f] for f in FEATURES}
        top_features = get_top_features_for_order(lgb_model, order_features, FEATURES)
        result = explain_order(order_features, sample["risk_score"], top_features)
        print("\n=== Real borderline example from test set ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))