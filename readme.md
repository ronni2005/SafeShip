# Return Risk Scorer
### Razorpay AI Risk Manager Hackathon — Returns/RTO Track

Predicts whether a Cash-on-Delivery (COD) order is likely to be **rejected at the door or returned**, before the merchant ships it — and tells the merchant exactly what to do about it.

---

## Problem

Indian e-commerce merchants lose real money to COD orders that get refused at delivery or returned right after — reverse logistics, repackaging, and courier costs eat into margins on every one of these, even when a "convenience fee" is charged. Existing solutions are largely rule-based (e.g. "block COD above ₹5000"). We built a **learned, explainable, financially-quantified** alternative.

## What This System Does

A merchant's checkout system sends order details to our API the moment an order is placed. The system:

1. **Scores** the order's rejection risk using a trained LightGBM model
2. **Explains** the score in plain English using an LLM — grounded strictly in the model's actual top features, not free-form guessing
3. **Suggests one action**: ship normally / verify address via WhatsApp / request partial deposit / request prepaid
4. **Flags uncertain cases for a human** instead of auto-acting, when the model is unsure or the LLM's output looks ungrounded
5. **Quantifies the ₹ impact** — money saved vs. money put at risk from false positives, on a real held-out test set

## Why This Direction

The hackathon's AI Risk Manager track lists "Return-risk scorer" as an explicit example direction under the Returns/RTO loss category. We picked this over fraud or chargebacks and went deep on one specific, high-friction Indian e-commerce problem: predicting COD rejection risk with signals that actually matter locally (address quality, city tier, order timing) rather than generic global fraud features.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌────────────────┐
│   Order      │────▶│   FastAPI     │────▶│   LightGBM       │────▶│  Risk score +   │
│  (checkout)  │     │  /predict     │     │  (risk_score)    │     │  top 3 drivers  │
└─────────────┘     └──────────────┘     └─────────────────┘     └───────┬────────┘
                                                                            │
                                                                            ▼
                                                                  ┌──────────────────┐
                                                                  │  Gemini LLM        │
                                                                  │  (explanation +    │
                                                                  │   suggested action)│
                                                                  └───────┬────────────┘
                                                                            │
                                                                            ▼
                                                          ┌──────────────────────────────┐
                                                          │  Hallucination validator +     │
                                                          │  confidence check              │
                                                          │  → needs_human_review flag     │
                                                          └───────┬──────────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌─────────────────────────┐
                                                        │  Streamlit Dashboard      │
                                                        │  - Check an Order          │
                                                        │  - Review Queue (HITL)     │
                                                        │  - ROI Dashboard           │
                                                        └─────────────────────────┘
```

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Model | LightGBM | Native categorical handling, feature importance, fast to tune on tabular data |
| Baseline | Logistic Regression | Sanity check / comparison — shows model selection wasn't arbitrary |
| Explainability | Gemini (`gemini-3.6-flash`) | Explains the model's decision in plain English — never makes the risk decision itself |
| Backend | FastAPI | Auto request validation, auto-generated docs at `/docs`, async-ready |
| Frontend | Streamlit | Fast to build, good for a 1-week timeline |
| Data | Synthetic (4,000 rows) | Full control over realistic Indian COD risk patterns |

## Key Differentiators

1. **Indian-context COD features** — address completeness, city tier, order timing, past rejection history — not generic global fraud features
2. **WhatsApp deposit-link generator** — turns a risk score into an actual recoverable action (a real `wa.me` link with a pre-filled confirmation message), not just a number
3. **Financial ROI breakdown** — money saved vs. money put at risk (false positives) vs. money still lost (missed cases), computed on the real held-out test set, with an interactive threshold slider
4. **Human-in-the-loop safety net** — the LLM only *explains* a decision the ML model already made. Orders get flagged for human review when either (a) the model's own confidence is borderline (risk score 40–65%), or (b) the LLM's explanation references something not present in the actual input features (hallucination check)

## Model Performance (Held-Out Test Set)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Rule-Based Baseline (COD + high-value/bad-address/repeat-rejector) | 62.3% | 52.1% | 56.8% |
| Logistic Regression | 41.8% | 80.6% | 55.1% |
| **LightGBM (main, tuned via 5-fold CV RandomizedSearch, 80 iterations)** | 56.3% | **62.4%** | **59.2%** |

LightGBM beats the rule-based baseline on every metric that matters here — F1 (59.2% vs. 56.8%) and, more decision-relevant for this business problem, recall: LightGBM catches 103 real risky orders on the test set vs. the rule's 86, a 17-order difference. Given that a missed risky order costs the full reverse-logistics loss while a false positive only costs a fraction of order value (see ROI Dashboard), we weight this recall advantage as the more meaningful result, even though the F1 gap alone would already be a fair claim on its own.

**Ranking quality (AUC)** — a metric a binary rule structurally cannot be scored on at all, since it only has one fixed operating point: LightGBM AUC-ROC 0.842, AUC-PR 0.596; Logistic Regression AUC-ROC 0.844 (essentially tied), AUC-PR 0.623 (still slightly ahead of LightGBM). We're reporting this as-is rather than only the numbers that favor our main model — Logistic Regression's calibration is genuinely competitive here, which is part of why we kept it as a real comparison point rather than a token baseline.

Decision threshold was selected using **only the training set**, then frozen and applied once to the test set — avoiding a data-leakage mistake we caught and fixed mid-project (see below). LightGBM's hyperparameters were found via an 80-iteration `RandomizedSearchCV` with 5-fold stratified cross-validation on the training set only (search space included tree depth/leaves, learning rate, subsampling, and L1/L2 regularization), not hand-picked.

Decision threshold was selected using **only the training set**, then frozen and applied once to the test set — avoiding a data-leakage mistake we caught and fixed mid-project (see below).

**Top predictive features:** order value, address completeness, past COD rejections, order hour, city tier, payment mode, item category.

## Financial Impact

At the tuned decision threshold (~0.85), the system shows a **net positive** financial impact on the test set. At lower thresholds, the false-positive cost (genuine customers wrongly asked for prepaid) outweighs the savings from correctly caught risky orders — the dashboard's interactive threshold slider (and an Auto-Optimize button) lets a merchant see and tune this trade-off directly, rather than trusting a single fixed cutoff.

## Limitations & Design Trade-offs

- **Synthetic data**: The dataset is generated with hand-designed, realistic risk rules rather than real Razorpay transactions. We used deterministic percentile-based labeling in the final version to reduce label noise — this makes the problem somewhat cleaner than real life, where two similar-looking orders don't always behave identically.
- **Assumed cost figures**: ROI calculations use estimated constants (₹150 reverse-logistics cost per rejected order, 30% of order value lost per wrongly-flagged genuine customer). These are reasonable planning assumptions, not measured real-world figures — a real deployment would calibrate these from a merchant's actual data.
- **City-tier as a risk signal**: `city_tier` is meant to capture delivery/logistics difficulty, not a socioeconomic judgment — but we noticed it could flag Tier 2/3 customers with an otherwise clean order history almost on tier alone. We reduced its weight in the model and lean on the BORDERLINE human-review tier as a safety net rather than removing the signal entirely. We see this as a genuine trade-off worth more work, not a solved problem.
- **LLM latency**: Gemini API calls can occasionally take 15-30+ seconds; the dashboard's request timeout is set generously to accommodate this.

## Challenges & What We Learned

**LightGBM initially underperformed our own baseline.** Our first tuned run had LightGBM scoring *worse* than plain Logistic Regression — a bigger, fancier model doing worse is a real signal, not a fluke to explain away. With only ~1,600-4,000 training rows, LightGBM's default settings (200 trees, depth 5) were overfitting. We cut it back to 80 trees, depth 3, added `min_child_samples=30`, and — separately — realized we were using the default 0.5 decision threshold on a dataset where only ~18% of orders are actually rejected. Tuning the threshold on the *training* set (see leakage note below) closed the gap and eventually pulled LightGBM ahead.

**The synthetic data's "prepaid = safe" signal was getting drowned by noise.** Early versions of the dataset generator applied random noise uniformly, which was large enough to blur the sharp real-world difference between COD and prepaid rejection rates. We fixed this by scaling the noise proportionally to the underlying risk score, and separately found a logic bug where the "prepaid orders can't be rejected at the door" discount was only being applied to the *base* risk score, before several other rules had already added risk on top — so prepaid orders were still inheriting city-tier and category risk they shouldn't have. Moving that discount to apply *last*, after all other rules, fixed it (COD reject rate ended up ~29% vs. prepaid ~4-5%, which is the kind of gap a model can actually learn from).

**Caught a genuine data-leakage bug in threshold selection.** We were originally picking the LightGBM decision threshold by looking at `precision_recall_curve` computed on the *test set* — meaning the "held-out" evaluation wasn't actually held out from that one decision. Fixed by selecting the threshold using only the training set's own predictions, then freezing it before touching the test set at all. The reported F1 dropped slightly after the fix — which is the correct, honest outcome, not a regression.

**Manually engineered interaction features taught us something about tree models, rather than helping.** We added `high_value_cod` and `risky_address_cod` (hand-built combinations we thought the model needed). Both came back with near-zero feature importance. That's not a failed feature — it demonstrated that LightGBM's sequential tree splits already discover these same interactions on their own; hand-crafted interaction features matter far more for linear models (like our Logistic Regression baseline) than for tree-based ones.

**Two bugs surfaced only once the full dashboard was running against a live model**, not in isolated testing: the ROI tab was silently using a generic 0.5 cutoff instead of the model's actual tuned ~0.60 threshold, understating the false-positive cost; and the Review Queue was showing duplicate entries for the same order because we were deduplicating on exact explanation text, and Gemini doesn't phrase the same order identically twice. Both were fixed — the ROI threshold is now a live, tunable slider instead of a hidden constant, and the queue dedupes on `order_id`.

**Our first rule-vs-ML comparison was accidentally unfair, and fixing it properly took two attempts.** We first wrote a rule-based baseline to prove ML beats existing rule-based tools — but used the exact same thresholds our own dataset generator uses to create the labels, giving the rule unfair insider knowledge. Realizing this, our first fix overcorrected the other way: we swapped in a deliberately weak, naive rule (a flat "block COD above ₹5000," taken directly from the brief's own example) — which made the gap look artificially large, and wasn't a fair test either. We reverted to a genuinely competent, hand-designed rule (COD + high value OR poor address OR repeat rejector — the kind an experienced ops analyst would actually write) and, instead of tilting the comparison, ran a proper 5-fold cross-validated `RandomizedSearchCV` over LightGBM's hyperparameters (80 iterations, covering tree depth, leaves, learning rate, subsampling, and L1/L2 regularization) — legitimate tuning on the training set only, never touching the test set. That produced a real, earned improvement: F1 59.2% vs. the rule's 56.8%, and — more decision-relevant — 103 real risky orders caught vs. the rule's 86.

**City-tier as a risk signal raised a fairness question we didn't fully resolve.** Testing a customer with zero rejection history from a Tier 3 city, we noticed they could land in the BORDERLINE review zone almost entirely because of their city tier, not their behavior. We reduced that feature's weight and rely on human review for borderline cases as a partial mitigation, but consider this an open trade-off rather than a solved one — see Limitations above.

## Running Locally

```bash
# 1. Generate data
python3 generate_dataset.py
python3 split_dataset.py

# 2. Train models
python3 models/train.py

# 3. Set up your Gemini API key in .env
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Start backend (Terminal 1)
python -m uvicorn backend.app:app --reload

# 5. Start dashboard (Terminal 2)
streamlit run frontend/dashboard.py
```

Visit `http://127.0.0.1:8000/docs` for the API, and the Streamlit URL printed in Terminal 2 for the dashboard.

## Project Structure

```
razorpay-rto-guard/
├── data/
│   ├── generate_dataset.py
│   ├── split_dataset.py
│   ├── orders.csv / train.csv / test.csv
├── models/
│   ├── train.py
│   ├── lightgbm_model.pkl / logreg_model.pkl
│   ├── feature_importance.csv / model_comparison.csv
├── backend/
│   ├── app.py              # FastAPI /predict endpoint
│   ├── llm_explainer.py    # LLM explanation + hallucination validator
├── frontend/
│   └── dashboard.py        # Streamlit dashboard (3 tabs)
├── requirements.txt
└── README.md
```