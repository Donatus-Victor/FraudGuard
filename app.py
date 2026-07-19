import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline, decision threshold, and expected column order.
    Cached so these only load once per app session, not on every rerun."""
    pipeline = pickle.load(open("fraud_pipeline.pkl", "rb"))
    threshold = pickle.load(open("threshold.pkl", "rb"))
    columns = pickle.load(open("model_columns.pkl", "rb"))
    return pipeline, threshold, columns


pipeline, threshold, columns = load_artifacts()

st.title("🛡️ Fraud Detection")
st.caption(f"Model decision threshold: {threshold:.2f}")

st.subheader("Transaction details")

col1, col2 = st.columns(2)

with col1:
    #home_country = st.text_input("Home country", " US", "UK", " CA", "unknown")
    home_country = st.selectbox("Home country", ["US", "UK", "CA", "unknown"])
    source_currency = st.text_input("Source currency", "USD")
    dest_currency = st.selectbox("Destination currency", ["USD", "CNY", "GBP", "CAD", "NGN", "INR", "PHP", "EUR", "MXN"])
    #dest_currency = st.text_input("Destination currency", 'USD','CNY', 'GBP', 'CAD', 'NGN', 'INR', 'PHP', 'EUR', 'MXN' )
    channel = st.selectbox("Channel", ["mobile", "web", "atm", "branch"])
    amount_src = st.number_input("Amount (source currency)", min_value=0.0, value=850.0)
    amount_usd = st.number_input("Amount (USD)", min_value=0.0, value=850.0)
    fee = st.number_input("Fee", min_value=0.0, value=12.50)
    new_device = st.checkbox("New device", value=True)
    location_mismatch = st.checkbox("Location mismatch", value=True)
    ip_risk_score = st.slider("IP risk score", 0.0, 1.0, 0.82)
    kyc_tier = st.selectbox("KYC tier", ["standard", "basic", "enhanced"])

with col2:
    account_age_days = st.number_input("Account age (days)", min_value=0, value=14)
    device_trust_score = st.slider("Device trust score", 0.0, 1.0, 0.21)
    chargeback_history_count = st.number_input("Chargeback history count", min_value=0, value=1)
    risk_score_internal = st.slider("Internal risk score", 0.0, 1.0, 0.77)
    txn_velocity_1h = st.number_input("Transaction velocity (1h)", min_value=0, value=4)
    txn_velocity_24h = st.number_input("Transaction velocity (24h)", min_value=0, value=11)
    corridor_risk = st.slider("Corridor risk", 0.0, 1.0, 0.65)
    hour = st.slider("Hour of day", 0, 23, 2)
    day_of_week = st.selectbox(
        "Day of week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=5,
    )
    is_weekend = 1 if day_of_week in ("Saturday", "Sunday") else 0

if st.button("Check transaction", type="primary"):
    transaction = {
        "home_country": home_country,
        "source_currency": source_currency,
        "dest_currency": dest_currency,
        "channel": channel,
        "amount_src": amount_src,
        "amount_usd": amount_usd,
        "fee": fee,
        "new_device": new_device,
        "location_mismatch": location_mismatch,
        "ip_risk_score": ip_risk_score,
        "kyc_tier": kyc_tier,
        "account_age_days": account_age_days,
        "device_trust_score": device_trust_score,
        "chargeback_history_count": chargeback_history_count,
        "risk_score_internal": risk_score_internal,
        "txn_velocity_1h": txn_velocity_1h,
        "txn_velocity_24h": txn_velocity_24h,
        "corridor_risk": corridor_risk,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
    }

    txn_df = pd.DataFrame([transaction], columns=columns)

    fraud_probability = pipeline.predict_proba(txn_df)[0, 1]
    is_fraud_pred = int(fraud_probability >= threshold)

    st.divider()
    st.metric("Fraud probability", f"{fraud_probability:.2%}")

    if is_fraud_pred:
        st.error(f"⚠️ Prediction: FRAUD (threshold = {threshold:.2f})")
    else:
        st.success(f"✅ Prediction: NOT FRAUD (threshold = {threshold:.2f})")
