# Return Risk Scorer — AI Risk Manager for COD Orders

**Solves a real retention problem for financial solutions provider companies for efficient two-way payment transactions.
The Core Issue: merchants lose trust when too many legitimate COD orders get rejected at the door. This system catches genuinely risky orders early, lets merchants *talk to* borderline ones, and keeps more orders flowing through the platform.**

---

## The Problem

**From a Merchant's View:** Cash-on-delivery orders are riskier — customers can refuse at the door. A rejected order costs reverse-logistics, refund hassle, and lost margin. Today, merchants either accept all orders (and eat the losses) or use crude rules ("Tier 3 = auto-reject"), accidentally blocking legitimate sales.

**From Companies View:** When merchants lose money to undetected risky orders, they blame the platform. When they lose *legitimate* sales to overly aggressive fraud rules, they switch competitors. Companies needs to *predict* which orders will actually fail AND *engage* customers to fix them — not just block orders. That keeps merchants on the platform.

**Approach:** AI that predicts *and* intervenes. A LightGBM model catches risky orders + per-order feature explanations so merchants understand *why*. Then, if an order is borderline, an AI-generated WhatsApp asks the customer to confirm address or payment — turning a risky order into a safe one before shipping.

---

## Architecture: Two Sides of AI

### 1. Predictive (Risk Score)
- **LightGBM classifier** trained on synthetic COD order data
- Inputs: payment mode, city tier, past rejections, order value, item category, **address quality** (parsed from actual address text, not a guessed number), order timing, weekend flag
- Output: risk probability (0-1), mapped to tiers (LOW < 0.40, BORDERLINE 0.40-0.65, HIGH > 0.65)
- **Why LightGBM:** Captures non-linear interactions (e.g., high-value + Tier 3 + late order = compounding risk) that simpler models miss

### 2. Generative (Customer Enrichment) — *Experimental*
- **Issue detection** (deterministic): identifies the *primary fixable issue* from the model's top features
  - Incomplete address? → Ask for house number + PIN
  - High-value COD? → Ask for confirmation
  - Repeat rejections? → Ask to re-verify address
- **Gemini LLM** generates a tailored WhatsApp message for that *specific* issue (not generic)
- **Merchant engagement:** System creates a pre-filled `wa.me` link with the message text. Merchant clicks → opens WhatsApp → customer receives the exact message in one tap. No manual typing, no generic templates.
- **Response parsing** (regex-based, simple): When customer replies in the simulator, extracts structured data (PIN, house number, confirmation phrases) from their free-text
- **Re-scoring** (experimental): Appends enriched data to original order, re-runs through the model. Early testing shows promise but needs validation on real data.
- **Merchant decision:** approve, request prepaid, or escalate

**Why this combination works:** The model identifies risk *patterns*, the LLM *explains and asks*, the regex parser *grounds* answers back to real data. No part blindly trusts the AI — the merchant always decides. The enrichment loop is early-stage; we include it to show the *pattern* of predict-explain-engage, but production deployment would need real WhatsApp API integration + actual customer data validation.

---

## The Dataset: Why Synthetic, and What We Learned

We don't have a publicly available **"COD orders rejected at door"** dataset. Real transaction data for such instances is proprietary. So built a synthetic one.

**Why not just use Kaggle?** Generic e-commerce datasets (Olist, UK Online Retail) have order history but *no label for doorstep rejection*. You can't train a supervised classifier without a target. Guessing labels would be pointless — the model would learn nothing.

**Approach:** Generate features realistically (lognormal order values, city-tier weights matching India's e-commerce split, fashion as top rejection category), then build a *causal risk formula* (not random labels). Start with 5% base risk, add risk for each red flag (past rejections +12%, Tier 3 city +10%, high-value COD +25%, poor address +15%), sample final outcome from that probability.

**Key honesty:** This is *synthetic* data, so reported the results as "proof of concept on controlled scenarios." Real-world performance would need real merchant data (future work). But the *methodology* — detecting issues, how validating LLM, handling the address parsing — all of that transfers.

---

## Model Results: The Honest Comparison

Compared three approaches on the same test set (800 orders):

| Approach | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|
| **Rule-Based Baseline** | 46.06% | 52.41% | 49.03% | — |
| **Logistic Regression** | 36.88% | 76.55% | 49.78% | 0.816 |
| **LightGBM (Main Model)** | 51.80% | 49.66% | **50.70%** | 0.809 |

**LightGBM now wins on F1.** It balances precision and recall better — catches risky orders (49.66% recall) while being more confident about its flags (51.80% precision, lowest false-positive rate).

**Why LightGBM over LogReg?** Beyond the F1 edge, LightGBM's per-order feature contributions (SHAP values) let us explain to merchants *why* this specific order is risky. LogReg gives you a probability; LightGBM shows you "order_value is the biggest driver (35% importance), followed by order_hour (25%), past_rejections (15%), etc." That's what merchants need to make confident decisions.

---

## Key Differentiators: AI Solving the Real Problem (4 Things We Got Right)

### 1. Address Parsing (Not Magic, Just Rigorous)
Started with a fake feature: `address_completeness` as a random number 0-100. That's lazy. Real fix: generate synthetic address strings (each component — house number, PIN, landmark — independently present/missing), then parse with *actual regex logic*.

Why this matters: The model trained on "incomplete addresses *cause delivery failures*" (true fact), not on invisible noise. The model learned order_value matters more than address (511 vs 179 importance score) — realistic, since merchants care most about high-value orders. Address completeness ranks #4, contributing meaningfully but not dominating. When a merchant sees "address_completeness: 45", they know exactly what's missing.

### 2. Smart Customer Engagement (Experimental)
Instead of: *"Hi, please confirm your order details"* (generic, awkward)

System sends: *"Hi! Just need your PIN code to finalize delivery. Reply with your complete address if possible"* (specific, actionable)

Deterministic issue detection + Gemini-generated message + one-click `wa.me` link. Merchant doesn't type anything — just clicks to send. This is a UX fix to the "I flagged your order as risky, now what?" problem. Early-stage feature, included to show the *pattern* of AI predicting, explaining, and engaging.

### 3. Hallucination Safety (Trust But Verify)
LLMs can invent things. We validate Gemini's explanation against the actual data:
- **Keyword check:** Did it mention "credit score" or "bank account"? (never saw those in the input.)
- **Numeric check:** Did it claim "5 past rejections" when the real value is 2?

If caught, we flag for human review. The merchant sees "AI flagged this, but the explanation looks wrong — you decide."

### 4. Merchant Dashboard (Not Just Model Output)
- **Tab 1:** Check a single order → see risk + explanation → send WhatsApp → re-score after customer reply → approve/request prepaid
- **Tab 2:** Review queue for borderline orders (manual override always possible)
- **Tab 3:** Live ROI calculator — drag the decision threshold, see money saved vs. false-positive costs in real time

---

## Problems Mid Way and Fixes

### 1. Data Leakage in Threshold Selection
**The bug:** We picked the optimal decision threshold using the *test set* probabilities. That's cheating — the model had "seen" the test data's right answers when choosing the threshold.

**The fix:** Use out-of-fold (OOF) cross-validation on the *train set only*. Fit the model on fold 1-4, predict on fold 5 (held out), repeat. Pick the threshold from those held-out predictions. Test set never touched until final grading.

**Why it mattered:** This shifted the optimal threshold from 0.5 to ~0.59. Using an in-sample threshold would have reported better numbers, but they'd be lies.

### 2. The Address Completeness Disaster
**The original sin:** `address_completeness = np.random.normal(75, 18)`. A number with no connection to any real address. It looked smart in the data schema, but it was fake.

**Red flag:** The model weighted it heavily (top 5 features), but we couldn't explain why. "Because it's random noise the model happened to latch onto" is not a business insight.

**The fix:** Build real address strings, parse them genuinely. Now when the model says "address is risky," we can show the merchant "because the address is missing the PIN code" (factual, actionable).

**Lesson:** Naming a feature is not the same as understanding it. We spent hours on this before realizing we'd skipped a layer of honesty.

### 3. Recall Bias Gone Wrong
**The mistake:** We optimized hyperparameters for pure recall (catch all risky orders, ignore false alarms). Result: flagged 57% of orders. Merchants would ask half their customers for deposits. Useless.

**The fix:** Use F-beta scoring (beta=1.0, balanced F1) instead of pure recall. Acknowledges that misses are costly, but doesn't ignore the cost of false alarms. Also, let the dashboard's threshold slider be where merchants can *choose* how aggressive to be, rather than baking one extreme into the model.

### 4. Gemini SDK Chaos
**The pain:** Started with `google-generativeai`, then saw `google-genai` (newer), tried mixing them. Ended up with three different API syntaxes in one file. 500 errors everywhere.

**The fix:** Stick with one stable SDK (`google.generativeai`). Add error handling so an API failure returns a sensible fallback, not a crash.

---

## Limitations & Future Work

- **Synthetic data:** Real-world performance unknown. Needs actual merchant transaction data + rejection outcomes.
- **Address parser:** Hand-written regex, not NLP. Misses typos, abbreviations, regional variations.
- **Enrichment flow (experimental):** WhatsApp re-engagement is simulated in the dashboard. Production would need:
  - Real WhatsApp Business API integration (not just wa.me links)
  - Actual webhook to receive customer replies
  - Validation that re-scored orders actually perform better in production
- **Hallucination validator:** Catches invented data sources and wrong numbers, but not subtle mischaracterizations.
- **Scalability:** No Docker, no multi-user auth, no database. Current dashboard is single-session. Production would need:
  - Persistent order queue (database)
  - Multi-merchant auth + isolation
  - Async job queue for re-scoring large batches
  - Monitoring + retraining pipeline

---

## How to Run

```bash
# Setup
python3 data/generate_dataset.py      # Build synthetic data
python3 data/split_dataset.py          # 80/20 train/test split
python3 models/train.py                # Train all 3 models, save artifacts

# Serve
# Terminal 1:
python -m uvicorn backend.app:app --reload

# Terminal 2:
streamlit run frontend/dashboard.py
```

**API docs:** `http://127.0.0.1:8000/docs`  
**Dashboard:** `http://localhost:8501`

---

## What This Proves

1. **Financial transaction provider companies can reduce merchant churn** by catching order risk early *and* helping fix it (not just blocking)
2. **Simple ML + simple AI works better than complex AI alone** — issue detection is deterministic, WhatsApp generation is Gemini, response parsing is regex. Each layer does one thing well.


This is a prototype, not production-ready yet. It shows the *pattern*: predict + explain + engage + decide. That pattern scales to real data.
