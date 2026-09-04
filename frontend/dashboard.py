import streamlit as st
import requests
import pandas as pd
import altair as alt
import urllib.parse

st.set_page_config(page_title="Return Risk Scorer", layout="wide")

API_URL = "http://127.0.0.1:8000"

st.markdown("# 📦 Return Risk Scorer")
st.markdown("*AI Risk Manager — COD Return/Rejection Risk Detector*")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Check an Order", "⚠️ Review Queue", "💰 ROI Dashboard"])

# Initialize session state for review queue
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []

# Helper function for unified risk colors & icons
def get_tier_badge(risk_tier: str):
    if risk_tier == "LOW":
        return "green", "🟢 LOW"
    elif risk_tier == "BORDERLINE":
        return "orange", "🟡 BORDERLINE"
    else:
        return "red", "🔴 HIGH"

# ============================================================================
# TAB 1: Check an Order (including enrichment flow)
# ============================================================================
with tab1:
    st.subheader("Enter Order Details")

    presets = {
        "Custom": None,
        "High-Value Fashion COD (Tier 3)": {
            "order_id": "ORD-DEMO-1", "is_prepaid": "COD", "city_tier": 3,
            "past_cod_rejections": 3, "order_value": 8500, "item_category": "fashion",
            "address_text": "Nehru Nagar, Ranchi", "order_hour": 23, "is_weekend": 1
        },
        "Low-Value Grocery Prepaid (Tier 1)": {
            "order_id": "ORD-DEMO-2", "is_prepaid": "Prepaid", "city_tier": 1,
            "past_cod_rejections": 0, "order_value": 600, "item_category": "grocery",
            "address_text": "Flat 4B, MG Road, Near City Hospital, Mumbai, 400001", "order_hour": 12, "is_weekend": 0
        },
        "Mid-Value Electronics (Tier 2)": {
            "order_id": "ORD-DEMO-3", "is_prepaid": "COD", "city_tier": 2,
            "past_cod_rejections": 1, "order_value": 3500, "item_category": "electronics",
            "address_text": "23, Station Road, Pune, 411001", "order_hour": 16, "is_weekend": 0
        },
    }
    preset_choice = st.selectbox("Quick preset (optional)", list(presets.keys()))
    p = presets[preset_choice]

    col1, col2 = st.columns(2)

    with col1:
        order_id = st.text_input("Order ID", value=p["order_id"] if p else "ORD00123")
        is_prepaid = st.selectbox("Payment Mode", ["COD", "Prepaid"], index=0 if not p or p["is_prepaid"] == "COD" else 1)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=(p["city_tier"] - 1) if p else 0)
        past_cod_rejections = st.number_input("Customer's Past COD Rejections", min_value=0, value=p["past_cod_rejections"] if p else 0)
        order_value = st.number_input("Order Value (₹)", min_value=1, value=p["order_value"] if p else 1500)

    with col2:
        categories = ["fashion", "electronics", "home", "beauty", "grocery", "accessories"]
        item_category = st.selectbox(
            "Item Category", categories, index=categories.index(p["item_category"]) if p else 0
        )
        address_text = st.text_area(
            "Delivery Address",
            value=p["address_text"] if p else "H.No. 8, MG Road, Mumbai",
            help="Type a real address — the system parses it for completeness (house number, PIN code, landmark) itself."
        )
        order_hour = st.slider("Order Hour (24h)", 0, 23, p["order_hour"] if p else 14)
        is_weekend = st.selectbox("Weekend Order?", ["No", "Yes"], index=(1 if p and p["is_weekend"] == 1 else 0))

    if st.button("🔎 Check Risk", type="primary"):
        payload = {
            "order_id": order_id,
            "is_prepaid": 1 if is_prepaid == "Prepaid" else 0,
            "city_tier": city_tier,
            "past_cod_rejections": past_cod_rejections,
            "order_value": float(order_value),
            "item_category": item_category,
            "address_text": address_text,
            "order_hour": order_hour,
            "is_weekend": 1 if is_weekend == "Yes" else 0,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the API. Make sure FastAPI is running: `python -m uvicorn backend.app:app --reload`")
            st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

        # Store result in session state
        st.session_state.last_prediction = result
        st.session_state.last_payload = payload

        # Add to human review queue if flagged by API
        if result.get("needs_human_review"):
            if not any(item["order_id"] == result["order_id"] for item in st.session_state.review_queue):
                st.session_state.review_queue.append(result)

        # Display risk score using API risk_tier dynamically
        risk_pct = result["risk_score"]
        risk_tier = result.get("risk_tier", "LOW")
        color, tier_badge = get_tier_badge(risk_tier)

        st.markdown(f"### Risk Score: :{color}[{risk_pct:.0%}] — :{color}[{tier_badge}]")
        st.caption(f"📍 Parsed address completeness: **{result.get('address_completeness', '—')}/100**")

        # Explanation and suggestion
        st.info(f"**Why:** {result['explanation']}")
        st.success(f"**Suggested Action:** {result['suggested_action']}")

        # Top risk drivers
        if result.get("top_features"):
            st.markdown("**Top factors driving this score:**")
            fcols = st.columns(len(result["top_features"]))
            for fc, (fname, fval) in zip(fcols, result["top_features"]):
                fc.metric(fname.replace("_", " ").title(), str(fval))

        st.divider()

        # ENRICHMENT FLOW (if BORDERLINE, HIGH, or flagged for review)
        if result.get("risk_tier") in ["BORDERLINE", "HIGH"] or result.get("needs_human_review"):
            st.subheader("📱 WhatsApp Enrichment")
            st.markdown("*The system detected a specific issue. Send a message to the customer to verify.*")

            # Show the issue
            st.warning(f"**Issue:** {result.get('issue_description', 'Needs verification')}")

            # Show the tailored WhatsApp message
            st.code(result.get("whatsapp_message", "Hi there! Please confirm your order details."), language="text")
            
            # Display WhatsApp Link Button
            wa_url = result.get("whatsapp_link", "")
            if wa_url:
                st.markdown(f"[📲 Open Pre-filled WhatsApp Message]({wa_url})", unsafe_allow_html=True)
            
            st.caption("👇 Simulate customer response below:")

            # Customer response input
            customer_response = st.text_area(
                "Simulate customer's WhatsApp reply (or leave blank to skip enrichment)",
                placeholder="e.g., 'H No 8, MG Road, Pincode 400001' or 'Yes, confirm this order'",
                height=60,
                key="customer_response"
            )

            if customer_response.strip():
                if st.button("✅ Enrich & Re-score", type="primary"):
                    try:
                        from backend.llm_explainer import parse_customer_response
                        issue_type = result.get("issue_type", "generic_verify")
                        parse_result = parse_customer_response(customer_response, issue_type)

                        if parse_result.get("success"):
                            st.success(f"✅ Enrichment successful! Extracted: {parse_result['extracted_data']}")

                            # Make a copy of the original order payload
                            enriched_payload = st.session_state.last_payload.copy()

                            # 1. Update the full address string with customer input
                            updated_address = f"{enriched_payload['address_text']}, {customer_response.strip()}"
                            enriched_payload["address_text"] = updated_address

                            # 2. Re-parse address to dynamically update completeness score
                            try:
                                from backend.address_parser import parse_address
                                parsed_addr = parse_address(updated_address)
                                enriched_payload["address_completeness"] = parsed_addr["completeness_score"]
                                st.info(f"📍 Address updated. New Completeness Score: **{parsed_addr['completeness_score']}/100**")
                            except ImportError:
                                st.info("Address string updated with customer input.")

                            # 3. Re-score order with updated payload
                            resp = requests.post(f"{API_URL}/predict", json=enriched_payload, timeout=45)
                            resp.raise_for_status()
                            enriched_result = resp.json()

                            e_pct = enriched_result["risk_score"]
                            e_tier = enriched_result.get("risk_tier", "LOW")
                            e_color, e_badge = get_tier_badge(e_tier)

                            st.markdown("---")
                            st.subheader("After Enrichment")
                            st.markdown(f"### Updated Risk: :{e_color}[{e_pct:.0%}] — :{e_color}[{e_badge}]")
                            st.info(f"**Updated Explanation:** {enriched_result['explanation']}")

                            st.divider()
                            st.subheader("Merchant Decision")
                            col_ship, col_prepaid = st.columns(2)

                            with col_ship:
                                if st.button("✅ Approve (Ship Order)", key="final_approve"):
                                    st.success(f"✅ Order {order_id} approved for shipping!")

                            with col_prepaid:
                                if st.button("💳 Request Prepaid/Deposit", key="final_prepaid"):
                                    deposit_amt = int(enriched_payload.get("order_value", 1000) * 0.10)
                                    msg = urllib.parse.quote(f"Hi! We need a deposit of ₹{deposit_amt} to confirm your order.")
                                    wa_link = f"https://wa.me/?text={msg}"
                                    st.markdown(f"[Send WhatsApp Deposit Link]({wa_link})", unsafe_allow_html=True)
                                    st.caption(f"Suggested deposit: ₹{deposit_amt} (10% of order value)")

                        else:
                            st.warning(f"❌ Couldn't fully parse response. Got: {parse_result.get('extracted_data')}\n\nAsk the customer for more details.")

                    except Exception as e:
                        st.error(f"Error re-scoring: {e}")

        # Final decision if low risk
        elif result.get("risk_tier") == "LOW":
            st.divider()
            st.subheader("Merchant Decision")
            if st.button("✅ Order is Safe — Approve for Shipping", key="low_risk_approve"):
                st.success(f"✅ Order {order_id} approved for immediate shipping!")

# ============================================================================
# TAB 2: Review Queue
# ============================================================================
with tab2:
    st.subheader("Orders Needing Human Review")
    st.markdown("*System flagged these as low-confidence or borderline risk — a human decides, the AI doesn't auto-act.*")

    if not st.session_state.review_queue:
        st.info("No orders in the review queue yet. Check an order in Tab 1 that lands in the borderline zone.")
    else:
        export_df = pd.DataFrame(st.session_state.review_queue)[
            ["order_id", "risk_score", "risk_tier", "explanation", "suggested_action", "review_reason"]
        ]
        st.download_button(
            "⬇️ Export Review Queue as CSV",
            data=export_df.to_csv(index=False),
            file_name="review_queue.csv",
            mime="text/csv",
        )

        for i, item in enumerate(st.session_state.review_queue):
            with st.container(border=True):
                _, badge = get_tier_badge(item.get('risk_tier', 'LOW'))
                st.markdown(f"**Order {item['order_id']} — Risk: {item['risk_score']:.0%} ({badge})**")
                st.markdown(f"*AI explanation:* {item['explanation']}")
                st.markdown(f"*Reason for review:* {item.get('review_reason', 'Flagged for review')}")

                col_approve, col_override = st.columns(2)
                with col_approve:
                    if st.button("✅ Approve (Ship)", key=f"approve_{i}"):
                        st.success("Order approved for shipping")
                        st.session_state.review_queue[i]["status"] = "approved"

                with col_override:
                    if st.button("🚫 Override (Block)", key=f"override_{i}"):
                        st.warning("Order blocked — request prepaid or cancel")
                        st.session_state.review_queue[i]["status"] = "blocked"

# ============================================================================
# TAB 3: ROI Dashboard
# ============================================================================
with tab3:
    st.subheader("Financial Impact Analysis")
    st.markdown("*See the money saved by catching risky orders early, vs. the cost of false alarms.*")

    # Persistent state toggle for ROI analysis
    if "roi_ran" not in st.session_state:
        st.session_state.roi_ran = False

    if st.button("Run ROI Analysis on Test Set"):
        st.session_state.roi_ran = True

    if st.session_state.roi_ran:
        try:
            # Cache test calculations in session state to prevent reload drops
            if "test_df_scored" not in st.session_state:
                test_df = pd.read_csv("data/test.csv")
                model = __import__("joblib").load("models/lightgbm_model.pkl")

                # Compute engineered interaction features
                test_df["high_value_cod"] = ((test_df["order_value"] > 3000) & (test_df["is_prepaid"] == 0)).astype(int)
                test_df["risky_address_cod"] = ((test_df["address_completeness"] < 70) & (test_df["is_prepaid"] == 0)).astype(int)

                FEATURES = [
                    "is_prepaid", "city_tier", "past_cod_rejections", "order_value", "item_category",
                    "address_completeness", "order_hour", "is_weekend", "high_value_cod", "risky_address_cod"
                ]
                rows = test_df[FEATURES].copy()
                rows["item_category"] = rows["item_category"].astype("category")
                test_df["risk_score"] = model.predict_proba(rows)[:, 1]
                st.session_state.test_df_scored = test_df

            test_df = st.session_state.test_df_scored

            REVERSE_LOGISTICS_COST = 150
            FALSE_POSITIVE_LOSS_RATE = 0.30

            if "roi_threshold" not in st.session_state:
                st.session_state.roi_threshold = 0.50

            col_slider, col_auto = st.columns([3, 1])
            with col_slider:
                st.markdown("**Decision threshold** (try adjusting to see trade-off live):")
            with col_auto:
                if st.button("⚡ Auto-Optimize"):
                    best_t, best_net = 0.50, float("-inf")
                    for t in [round(x * 0.01, 2) for x in range(10, 96, 1)]:
                        pred = (test_df["risk_score"] >= t).astype(int)
                        tp = ((pred == 1) & (test_df["rejected"] == 1)).sum()
                        fp_mask = (pred == 1) & (test_df["rejected"] == 0)
                        net = tp * REVERSE_LOGISTICS_COST - (test_df.loc[fp_mask, "order_value"] * FALSE_POSITIVE_LOSS_RATE).sum()
                        if net > best_net:
                            best_net, best_t = net, t
                    st.session_state.roi_threshold = best_t
                    st.toast(f"Optimal threshold found: {best_t} (Net: ₹{best_net:,.0f})")
                    st.rerun()

            threshold = st.slider(
                "Risk score above this = flagged as risky", 0.0, 1.0,
                st.session_state.roi_threshold, 0.01,
                help="Adjust to see cost trade-offs.",
                key="roi_threshold"
            )
            test_df["predicted_risky"] = (test_df["risk_score"] >= threshold).astype(int)

            # Metrics calculation
            tp = ((test_df["predicted_risky"] == 1) & (test_df["rejected"] == 1)).sum()
            fp_mask = (test_df["predicted_risky"] == 1) & (test_df["rejected"] == 0)
            fp = fp_mask.sum()
            fn = ((test_df["predicted_risky"] == 0) & (test_df["rejected"] == 1)).sum()

            money_saved = tp * REVERSE_LOGISTICS_COST
            money_at_risk = (test_df.loc[fp_mask, "order_value"] * FALSE_POSITIVE_LOSS_RATE).sum()
            money_still_lost = fn * REVERSE_LOGISTICS_COST
            net_benefit = money_saved - money_at_risk

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Money Saved", f"₹{money_saved:,.0f}", help="Reverse-logistics cost avoided")
            col2.metric("⚠️ Money At Risk", f"₹{money_at_risk:,.0f}", help="Lost sale + friction from false alarms")
            col3.metric("❌ Still Lost", f"₹{money_still_lost:,.0f}", help="Missed risky orders")
            col4.metric("✅ Net Benefit", f"₹{net_benefit:,.0f}", delta=f"{(net_benefit / (money_saved + 1)):+.0%}")

            # Chart representation
            chart_df = pd.DataFrame({
                "Category": ["Saved", "At risk (wrongly flagged)", "Missed (still lost)"],
                "Amount": [money_saved, money_at_risk, money_still_lost],
                "Color": ["#2ecc71", "#e74c3c", "#f39c12"]
            })
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Category:N", sort=None, axis=alt.Axis(labelAngle=0, labelLimit=200)),
                y=alt.Y("Amount:Q", title="Amount (₹)"),
                color=alt.Color("Color:N", scale=None, legend=None),
                tooltip=["Category", "Amount"]
            ).properties(height=350)
            st.altair_chart(chart, width="stretch")

        except Exception as e:
            st.error(f"Error running ROI: {e}")