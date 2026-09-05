"""
Return Risk Scorer — Synthetic Dataset
~4000 realistic e-commerce orders (COD + Prepaid mix)
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from address_scorer import score_address_completeness

np.random.seed(42)  
N = 4000

# STEP 1: Generate raw feature columns

# Payment mode: 55% COD, 45% Prepaid — India still has heavy COD share
is_prepaid = np.random.choice([0, 1], size=N, p=[0.55, 0.45])

# City tier: Tier 1 (metros) = lower reject risk, Tier 2/3 = higher
city_tier = np.random.choice([1, 2, 3], size=N, p=[0.30, 0.45, 0.25])

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
    p=[0.30, 0.20, 0.10, 0.15, 0.10, 0.15]
)

# Order hour (0-23) and weekend flag
order_hour = np.random.randint(0, 24, size=N)
is_weekend = np.random.choice([0, 1], size=N, p=[0.7, 0.3])

# STEP 1b: Generate a REAL synthetic address string per order,

STREETS = ["MG Road", "Nehru Nagar", "Gandhi Marg", "Station Road", "Lake View Lane",
           "Church Street", "Model Colony", "Sector 12", "Ashok Vihar", "Park Street"]
LANDMARKS = ["Near City Hospital", "Opposite SBI Bank", "Behind Central Mall",
             "Near Govt School", "Opposite Petrol Pump", "Near Water Tank"]
CITIES = ["Mumbai", "Pune", "Lucknow", "Patna", "Ranchi", "Nagpur", "Indore",
          "Guwahati", "Bhopal", "Kanpur", "Mirzapur", "Jaipur"]


def build_address(has_house_no, has_street, has_landmark, has_pin, city):
    """Constructs a realistic address string with the chosen components present/missing."""
    parts = []
    if has_house_no:
        parts.append(np.random.choice(["H.No. 8", "Flat 4B", "23", "Plot No. 17", "Door No. 5"]))
    parts.append(np.random.choice(STREETS))  # street name itself always attempted
    if has_landmark:
        parts.append(np.random.choice(LANDMARKS))
    parts.append(city)
    if has_pin:
        parts.append(str(np.random.randint(110000, 855999)))  # 6-digit Indian PIN
    return ", ".join(parts)


has_house_no = np.random.choice([0, 1], size=N, p=[0.15, 0.85])
has_street = np.ones(N, dtype=int) 
has_landmark = np.random.choice([0, 1], size=N, p=[0.55, 0.45])
has_pin = np.random.choice([0, 1], size=N, p=[0.25, 0.75])
address_city = np.random.choice(CITIES, size=N)

address_text = [
    build_address(has_house_no[i], has_street[i], has_landmark[i], has_pin[i], address_city[i])
    for i in range(N)
]
address_completeness = np.array([score_address_completeness(a) for a in address_text])

# STEP 2: Build reject probability using real-world logic

risk_score = np.full(N, 0.03)  # base 5% reject chance for everyone

# Rule 2: Higher city tier number (2/3) = higher reject risk
risk_score += np.where(city_tier == 2, 0.07, 0)
risk_score += np.where(city_tier == 3, 0.10, 0)

# Rule 3: Past rejections are the STRONGEST predictor — repeat behavior
risk_score += past_cod_rejections * 0.12

# Rule 4: High order value on COD = cold feet risk
risk_score += np.where((order_value > 3000) & (is_prepaid == 0), 0.25, 0)
risk_score += np.where((address_completeness < 50) & (is_prepaid == 0), 0.30, 0)  # extra bump

# Rule 5: Category effect — fashion/electronics reject more (fit/quality doubts)
category_risk = {
    "fashion": 0.08, "electronics": 0.05 ,"accessories": 0.04,
    "home": 0.01, "beauty": 0.01, "grocery": -0.03
}
risk_score += pd.Series(item_category).map(category_risk).values

# Rule 6: Late-night impulse orders (11PM-4AM) reject more often
risk_score += np.where((order_hour >= 23) | (order_hour <= 4), 0.07, 0)

# Rule 8: Poor address completeness = higher failed-delivery/reject risk.
risk_score += np.where(address_completeness < 40, 0.15, 0)
risk_score += np.where((address_completeness >= 40) & (address_completeness < 75), 0.05, 0)

# Rule 7: Weekend orders slightly more impulsive
risk_score += np.where(is_weekend == 1, 0.03, 0)

# Noise scales with the risk itself so it doesn't drown out the strong
# "prepaid = safe" signal for low-risk orders.
noise = np.random.normal(0, 0.02, size=N) + risk_score * np.random.normal(0, 0.15, size=N)
risk_score += noise

# Rule 1 (applied LAST): Prepaid orders can't be "rejected at the door" —there's no COD refusal moment, so squash the whole accumulated risk.
risk_score = np.where(is_prepaid == 1, risk_score * 0.15, risk_score)

# Clip probability between 0 and 1
risk_score = np.clip(risk_score, 0.01, 0.95)

# STEP 3: Convert probability into actual 0/1 outcome
rejected = np.random.binomial(1, risk_score)

# STEP 4: Assemble final dataframe
df = pd.DataFrame({
    "order_id": [f"ORD{i:05d}" for i in range(1, N + 1)],
    "is_prepaid": is_prepaid,
    "city_tier": city_tier,
    "past_cod_rejections": past_cod_rejections,
    "order_value": order_value,
    "item_category": item_category,
    "address_text": address_text,             
    "address_completeness": address_completeness,
    "order_hour": order_hour,
    "is_weekend": is_weekend,
    "rejected": rejected 
})

df.to_csv("data/orders.csv", index=False)

print(f"Generated {N} rows -> data/orders.csv")
print(f"Overall reject rate: {df['rejected'].mean():.1%}")
print(f"COD reject rate: {df[df.is_prepaid==0]['rejected'].mean():.1%}")
print(f"Prepaid reject rate: {df[df.is_prepaid==1]['rejected'].mean():.1%}")
print(f"\nAddress completeness distribution:\n{df['address_completeness'].describe()}")
print("\nSample addresses and their parsed scores:")
print(df[["address_text", "address_completeness"]].head(8).to_string(index=False))