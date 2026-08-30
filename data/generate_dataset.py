"""
Return Risk Scorer — Synthetic Dataset Generator
==================================================
Generates ~2000 realistic e-commerce orders (COD + Prepaid mix)
with a REJECTED/ACCEPTED label that follows real-world logic,
not random noise.

WHY THIS MATTERS:
A model is only as good as the patterns in its training data.
If we randomly assign "rejected" labels, the model learns nothing.
So each feature below nudges the reject probability up or down,
the way it would in real life. This is what makes the dataset
"synthetic but realistic" instead of "randomly generated."
"""

import pandas as pd
import numpy as np

np.random.seed(42)  # reproducibility — same data every time we run this

N = 4000

# ---------------------------------------------------------
# STEP 1: Generate raw feature columns
# ---------------------------------------------------------

# Payment mode: 55% COD, 45% Prepaid — India still has heavy COD share
is_prepaid = np.random.choice([0, 1], size=N, p=[0.55, 0.45])

# City tier: Tier 1 (metros) = lower reject risk, Tier 2/3 = higher
city_tier = np.random.choice([1, 2, 3], size=N, p=[0.35, 0.4, 0.25])

# Past COD rejections by this customer (most people have 0, few are repeat offenders)
past_cod_rejections = np.random.choice(
    [0, 1, 2, 3, 4, 5],
    size=N,
    p=[0.55, 0.2, 0.12, 0.07, 0.04, 0.02]
)

# Order value in ₹ (realistic e-commerce spread, right-skewed)
order_value = np.round(np.random.lognormal(mean=6.5, sigma=0.6, size=N), -1)
order_value = np.clip(order_value, 200, 15000)

# Item category — some categories get "tried and rejected" more (fashion/shoes)
item_category = np.random.choice(
    ["fashion", "electronics", "home", "beauty", "grocery", "accessories"],
    size=N,
    p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
)

# Address completeness score (0-100): landmark present, correct PIN, flat/house
# number filled, etc. Lower score = courier struggles to find/deliver =
# higher chance of failed delivery being logged as a "reject"
address_completeness = np.clip(
    np.round(np.random.normal(75, 18, size=N)), 20, 100
)

# Order hour (0-23) and weekend flag
order_hour = np.random.randint(0, 24, size=N)
is_weekend = np.random.choice([0, 1], size=N, p=[0.7, 0.3])

# ---------------------------------------------------------
# STEP 2: Build reject probability using real-world logic
# ---------------------------------------------------------
# We start with a small base risk, then ADD risk for each red flag.
# This mimics how a real merchant/risk analyst would reason.

risk_score = np.full(N, 0.05)  # base 5% reject chance for everyone

# Rule 2: Higher city tier number (2/3) = higher reject risk
risk_score += np.where(city_tier == 2, 0.08, 0)
risk_score += np.where(city_tier == 3, 0.10, 0)

# Rule 3: Past rejections are the STRONGEST predictor — repeat behavior
risk_score += past_cod_rejections * 0.12

# Rule 4: High order value on COD = cold feet risk
risk_score += np.where((order_value > 3000) & (is_prepaid == 0), 0.25, 0)
risk_score += np.where((address_completeness < 50) & (is_prepaid == 0), 0.30, 0)  # extra bump

# Rule 5: Category effect — fashion/electronics reject more (fit/quality doubts)
category_risk = {
    "fashion": 0.08, "electronics": 0.06, "accessories": 0.03,
    "home": 0.01, "beauty": 0.01, "grocery": -0.03
}
risk_score += pd.Series(item_category).map(category_risk).values

# Rule 6: Late-night impulse orders (11PM-4AM) reject more often
risk_score += np.where((order_hour >= 23) | (order_hour <= 4), 0.07, 0)

# Rule 8: Poor address completeness = higher failed-delivery/reject risk
# Score < 50 = "incomplete", strong risk bump; 50-75 = mild bump; 75+ = safe
risk_score += np.where(address_completeness < 50, 0.15, 0)
risk_score += np.where((address_completeness >= 50) & (address_completeness < 75), 0.05, 0)

# Rule 7: Weekend orders slightly more impulsive
risk_score += np.where(is_weekend == 1, 0.03, 0)

# Add small random noise so it's not a perfectly deterministic formula
# (real life always has some unexplainable randomness).
# Noise scales with the risk itself so it doesn't drown out the strong
# "prepaid = safe" signal for low-risk orders.
noise = np.random.normal(0, 0.02, size=N) + risk_score * np.random.normal(0, 0.15, size=N)
risk_score += noise

# Rule 1 (applied LAST): Prepaid orders can't be "rejected at the door" —
# there's no COD refusal moment, so squash the whole accumulated risk down.
risk_score = np.where(is_prepaid == 1, risk_score * 0.15, risk_score)

# Clip probability between 0 and 1
risk_score = np.clip(risk_score, 0.01, 0.95)

# ---------------------------------------------------------
# STEP 3: Convert probability into actual 0/1 outcome
# ---------------------------------------------------------
rejected = np.random.binomial(1, risk_score)

# ---------------------------------------------------------
# STEP 4: Assemble final dataframe
# ---------------------------------------------------------
df = pd.DataFrame({
    "order_id": [f"ORD{i:05d}" for i in range(1, N + 1)],
    "is_prepaid": is_prepaid,
    "city_tier": city_tier,
    "past_cod_rejections": past_cod_rejections,
    "order_value": order_value,
    "item_category": item_category,
    "address_completeness": address_completeness,
    "order_hour": order_hour,
    "is_weekend": is_weekend,
    "rejected": rejected  # <-- this is our TARGET / label
})

df.to_csv("data/orders.csv", index=False)

print(f"Generated {N} rows -> data/orders.csv")
print(f"Overall reject rate: {df['rejected'].mean():.1%}")
print(f"COD reject rate: {df[df.is_prepaid==0]['rejected'].mean():.1%}")
print(f"Prepaid reject rate: {df[df.is_prepaid==1]['rejected'].mean():.1%}")
print("\nSample rows:")
print(df.head())