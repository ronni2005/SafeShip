"""
Model Training — Logistic Regression (baseline) vs LightGBM (main)
=====================================================================
WHY TWO MODELS:
- Logistic Regression = simple, interpretable, fast. Our "sanity check".
- LightGBM = handles non-linear patterns + feature interactions much
  better (e.g. "high value AND COD AND late night" combo risk).
We train both on the SAME train.csv, evaluate on the SAME test.csv,
so the comparison is fair.

WHY PRECISION & RECALL (not just accuracy):
Only ~18% of orders are actually rejected. A model that predicts
"never risky" would be 82% "accurate" but catch zero real risk —
useless. So we care about:
  - Precision: of orders we FLAG as risky, how many really were?
    (low precision = annoying genuine customers for no reason)
  - Recall: of orders that WERE actually rejected, how many did we catch?
    (low recall = missing real fraud/return risk = money lost)
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import joblib

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Add Interaction Features (Exact logic from your image)
for df in [train_df, test_df]:
    df['high_value_cod'] = ((df['order_value'] > 3000) & (df['is_prepaid'] == 0)).astype(int)
    
    # Note: If address_completeness is a float (0.3 to 1.0), use 0.50 instead of 50
    addr_threshold = 0.50 if train_df['address_completeness'].max() <= 1.0 else 50
    df['risky_address_cod'] = ((df['address_completeness'] < addr_threshold) & (df['is_prepaid'] == 0)).astype(int)
FEATURES = ["is_prepaid", "city_tier", "past_cod_rejections",
            "order_value", "item_category", "address_completeness",
            "order_hour", "is_weekend", "high_value_cod", "risky_address_cod"]
TARGET = "rejected"

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

NUMERIC = ["is_prepaid", "city_tier", "past_cod_rejections", "order_value",
           "address_completeness", "order_hour", "is_weekend", "high_value_cod", "risky_address_cod"]
CATEGORICAL = ["item_category"]


def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n--- {name} ---")
    print(f"Precision: {p:.2%}  |  Recall: {r:.2%}  |  F1: {f1:.2%}")
    print(f"Caught {tp} real risky orders, missed {fn}, wrongly flagged {fp} safe ones")
    return {"model": name, "precision": p, "recall": r, "f1": f1,
            "true_positive": int(tp), "false_positive": int(fp), "false_negative": int(fn)}


# ---------------------------------------------------------
# MODEL 1: Logistic Regression (needs scaled numeric + one-hot categorical)
# ---------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])

logreg_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])
# class_weight="balanced" matters here — since only ~18% of orders are
# rejected, without this the model leans toward always predicting "safe"

logreg_pipeline.fit(X_train, y_train)
logreg_preds = logreg_pipeline.predict(X_test)
logreg_results = evaluate("Logistic Regression (baseline)", y_test, logreg_preds)

# ---------------------------------------------------------
# MODEL 2: LightGBM (handles categorical natively, no encoding needed)
# ---------------------------------------------------------
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
X_train_lgb["item_category"] = X_train_lgb["item_category"].astype("category")
X_test_lgb["item_category"] = X_test_lgb["item_category"].astype("category")

lgb_model = lgb.LGBMClassifier(
    class_weight="balanced",
    random_state=42,
    n_estimators=80,
    max_depth=3,
    num_leaves=15,
    min_child_samples=30,
    learning_rate=0.08,
    verbose=-1
)
lgb_model.fit(X_train_lgb, y_train, categorical_feature=["item_category"])

# Instead of the default 0.5 cutoff, find the threshold that best balances
# precision & recall on this imbalanced data (~18% positive class).
from sklearn.metrics import precision_recall_curve
probs_train = lgb_model.predict_proba(X_train_lgb)[:, 1]
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_train, probs_train)
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_idx = f1_arr[:-1].argmax()
best_threshold = thresh_arr[best_idx]
print(f"\nBest decision threshold found: {best_threshold:.3f} (instead of default 0.5)")

probs_test=lgb_model.predict_proba(X_test_lgb)[:,1]
lgb_preds = (probs_test >= best_threshold).astype(int)
lgb_results = evaluate("LightGBM (main model)", y_test, lgb_preds)

# ---------------------------------------------------------
# Feature importance — needed later for the LLM explainer (Day 3)
# ---------------------------------------------------------
importance = pd.DataFrame({
    "feature": X_train_lgb.columns,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n--- LightGBM Feature Importance ---")
print(importance.to_string(index=False))

# ---------------------------------------------------------
# Save everything Day 3/4 will need
# ---------------------------------------------------------
joblib.dump(lgb_model, "models/lightgbm_model.pkl")
joblib.dump(logreg_pipeline, "models/logreg_model.pkl")
importance.to_csv("models/feature_importance.csv", index=False)

results_df = pd.DataFrame([logreg_results, lgb_results])
results_df.to_csv("models/model_comparison.csv", index=False)
print("\nSaved: models/lightgbm_model.pkl, logreg_model.pkl, feature_importance.csv, model_comparison.csv")