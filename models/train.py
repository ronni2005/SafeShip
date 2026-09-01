"""
Model Training — Rule-Based Baseline vs Logistic Regression vs LightGBM
=========================================================================
WHY THREE COMPARISONS:
- Rule-Based = what a merchant ops team would hand-write today (static
  thresholds, no learning). This is the bar the hackathon brief says
  existing tools are stuck at — we need to beat it, provably.
- Logistic Regression = simple, interpretable, fast. Our ML "sanity check".
- LightGBM = handles non-linear patterns + feature interactions much
  better (e.g. "high value AND COD AND late night" combo risk).
All three are evaluated on the SAME test.csv, so the comparison is fair.

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


def evaluate_ranking(name, y_true, y_scores):
    """
    AUC-ROC / AUC-PR measure how well a model RANKS orders by risk across
    every possible threshold at once — not just one fixed cutoff. A binary
    rule (e.g. "block COD above Rs 5000: yes/no") has exactly ONE operating
    point, so it cannot be meaningfully scored this way at all. This is a
    structural capability gap a rule cannot close no matter how well its
    one threshold is tuned.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    print(f"{name} — AUC-ROC: {auc_roc:.3f}  |  AUC-PR: {auc_pr:.3f}")
    return auc_roc, auc_pr


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
# MODEL 0: Rule-Based Baseline — what most existing merchant tools do today
# ---------------------------------------------------------
# WHY THIS MATTERS: the hackathon brief explicitly says existing risk
# tools are "rule-based" and asks for a genuinely ML-driven improvement.
# Without this baseline, "our ML model is better than rules" is just an
# assertion. This makes it a measured, provable claim.
#
# This rule is a genuinely competent one, close to what an experienced
# ops analyst would write by hand: COD + (high value OR poor address OR
# repeat rejector). We're deliberately NOT weakening this to make our
# model look better — a fair comparison against a good rule is the only
# comparison worth reporting.
def rule_based_predict(df):
    return (
        (df["is_prepaid"] == 0) &
        ((df["order_value"] > 3000) | (df["address_completeness"] < 50) | (df["past_cod_rejections"] >= 2))
    ).astype(int)

rule_preds = rule_based_predict(test_df)
rule_results = evaluate("Rule-Based Baseline (COD + high-value/bad-address/repeat-rejector)", y_test, rule_preds)

# ---------------------------------------------------------
# MODEL 2: LightGBM — proper hyperparameter search (not hand-guessed)
# ---------------------------------------------------------
# Earlier versions of this model used manually-picked hyperparameters
# (n_estimators=80, max_depth=3, etc.) based on trial and error. This
# does a real RandomizedSearchCV instead — cross-validated on the
# TRAINING set only (never touches X_test/y_test), so it's a legitimate
# tuning process, not test-set peeking.
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
X_train_lgb["item_category"] = X_train_lgb["item_category"].astype("category")
X_test_lgb["item_category"] = X_test_lgb["item_category"].astype("category")

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

param_dist = {
    "n_estimators": [60, 80, 100, 150, 200, 300],
    "max_depth": [3, 4, 5, 6, 8, -1],
    "num_leaves": [10, 15, 20, 31, 40, 63],
    "min_child_samples": [10, 20, 30, 50, 70],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],       # L1 regularization — can reduce overfitting
    "reg_lambda": [0, 0.01, 0.1, 0.5, 1.0],      # L2 regularization
    "class_weight": ["balanced", None],           # let CV decide if class balancing even helps
}

base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=80,
    scoring="f1", cv=cv, random_state=42, n_jobs=-1
)
search.fit(X_train_lgb, y_train, categorical_feature=["item_category"])
lgb_model = search.best_estimator_
print(f"\nBest CV F1 (5-fold, train set only): {search.best_score_:.2%}")
print(f"Best hyperparameters: {search.best_params_}")

# Instead of the default 0.5 cutoff, find the threshold that best balances
# precision & recall — using ONLY the training set's own predictions.
# We must NOT look at y_test here: choosing a threshold based on test labels
# is a subtle form of data leakage (the threshold would be "tuned" to the
# very data we're trying to honestly evaluate on). The model already saw
# X_train/y_train during fit(), so using them again to pick a threshold
# introduces no NEW leakage — but touching y_test at this stage would.
from sklearn.metrics import precision_recall_curve
probs_train = lgb_model.predict_proba(X_train_lgb)[:, 1]
prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_train, probs_train)
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
best_threshold = thresh_arr[f1_arr[:-1].argmax()]
print(f"\nBest decision threshold found on TRAIN set: {best_threshold:.3f} (instead of default 0.5)")

# Threshold is now frozen. Test set is touched for the first and only time
# right here, purely for final honest evaluation.
probs_test = lgb_model.predict_proba(X_test_lgb)[:, 1]
lgb_preds = (probs_test >= best_threshold).astype(int)
lgb_results = evaluate("LightGBM (main model)", y_test, lgb_preds)

# ---------------------------------------------------------
# Ranking capability — the structural gap a rule cannot close
# ---------------------------------------------------------
print("\n--- Ranking Quality (AUC) — rules cannot be scored this way at all ---")
lgb_auc_roc, lgb_auc_pr = evaluate_ranking("LightGBM", y_test, probs_test)
logreg_probs_test = logreg_pipeline.predict_proba(X_test)[:, 1]
logreg_auc_roc, logreg_auc_pr = evaluate_ranking("Logistic Regression", y_test, logreg_probs_test)

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

results_df = pd.DataFrame([rule_results, logreg_results, lgb_results])
results_df.to_csv("models/model_comparison.csv", index=False)
print("\nSaved: models/lightgbm_model.pkl, logreg_model.pkl, feature_importance.csv, model_comparison.csv")