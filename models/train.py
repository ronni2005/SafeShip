"""
Model Training — Rule-Based Baseline vs Logistic Regression vs LightGBM
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
import lightgbm as lgb
import joblib

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# 2. Add Interaction Features
for df in [train_df, test_df]:
    df['high_value_cod'] = ((df['order_value'] > 3000) & (df['is_prepaid'] == 0)).astype(int)

    addr_threshold = 0.70 if train_df['address_completeness'].max() <= 1.0 else 70
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
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)
    print(f"{name} — AUC-ROC: {auc_roc:.3f}  |  AUC-PR: {auc_pr:.3f}")
    return auc_roc, auc_pr


# MODEL 1: Logistic Regression (UNTOUCHED)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])

logreg_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])

logreg_pipeline.fit(X_train, y_train)
logreg_preds = logreg_pipeline.predict(X_test)
logreg_results = evaluate("Logistic Regression (baseline)", y_test, logreg_preds)

# MODEL 0: Rule-Based Baseline (UNTOUCHED)
def rule_based_predict(df):
    return (
        (df["is_prepaid"] == 0) &
        ((df["order_value"] > 3000) | (df["address_completeness"] < 70) | (df["past_cod_rejections"] >= 2))
    ).astype(int)

rule_preds = rule_based_predict(test_df)
rule_results = evaluate("Rule-Based Baseline (COD + high-value/bad-address/repeat-rejector)", y_test, rule_preds)

# ==============================================================================
# MODEL 2: LightGBM (OPTIMIZED FOR HIGH RECALL & F1)
# ==============================================================================
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
X_train_lgb["item_category"] = X_train_lgb["item_category"].astype("category")
X_test_lgb["item_category"] = X_test_lgb["item_category"].astype("category")

# Calculate base imbalance ratio 
pos_weight_baseline = (len(y_train) - sum(y_train)) / sum(y_train)

param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "num_leaves": [7, 15, 25, 31],
    "min_child_samples": [5, 10, 15, 25],
    "learning_rate": [0.01, 0.03, 0.05, 0.08],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0.01, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda": [0.01, 0.1, 0.5, 1.0, 2.0],
    "scale_pos_weight": [2.0, 3.0, pos_weight_baseline, 5.0, 6.0],  # Fine-grain oversampling ratio
}

base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    base_model, param_distributions=param_dist, n_iter=100,
    scoring="f1", cv=cv, random_state=42, n_jobs=-1
)
search.fit(X_train_lgb, y_train, categorical_feature=["item_category"])
lgb_model = search.best_estimator_
print(f"\nBest CV F1 (5-fold, train set only): {search.best_score_:.2%}")
print(f"Best hyperparameters: {search.best_params_}")

# OPTIMIZATION: Use Out-Of-Fold (OOF) CV predictions to select threshold
# Prevents training set threshold leakage & artificially high cutoff values
oof_probs = cross_val_predict(
    lgb_model, X_train_lgb, y_train, cv=cv, method="predict_proba"
)[:, 1]

prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_train, oof_probs)

# Target metric optimization: Maximize F1 while keeping recall prioritized
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)

# Select best threshold index based on OOF F1
best_idx = f1_arr[:-1].argmax()
best_threshold = thresh_arr[best_idx]

print(f"\nOptimal Decision Threshold (via Out-Of-Fold CV): {best_threshold:.3f}")
print(f"Expected OOF Precision: {prec_arr[best_idx]:.2%} | Expected OOF Recall: {rec_arr[best_idx]:.2%}")

# Evaluate on test set
probs_test = lgb_model.predict_proba(X_test_lgb)[:, 1]
lgb_preds = (probs_test >= best_threshold).astype(int)
lgb_results = evaluate("LightGBM (main model)", y_test, lgb_preds)

# Ranking model performance
print("\n--- Ranking Quality (AUC) ---")
lgb_auc_roc, lgb_auc_pr = evaluate_ranking("LightGBM", y_test, probs_test)
logreg_probs_test = logreg_pipeline.predict_proba(X_test)[:, 1]
logreg_auc_roc, logreg_auc_pr = evaluate_ranking("Logistic Regression", y_test, logreg_probs_test)

# Feature importance
importance = pd.DataFrame({
    "feature": X_train_lgb.columns,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n--- LightGBM Feature Importance ---")
print(importance.to_string(index=False))

# Saving artifacts
joblib.dump(lgb_model, "models/lightgbm_model.pkl")
joblib.dump(logreg_pipeline, "models/logreg_model.pkl")
importance.to_csv("models/feature_importance.csv", index=False)

results_df = pd.DataFrame([rule_results, logreg_results, lgb_results])
results_df.to_csv("models/model_comparison.csv", index=False)
print("\nSaved: models/lightgbm_model.pkl, logreg_model.pkl, feature_importance.csv, model_comparison.csv")