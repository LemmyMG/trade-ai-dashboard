import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------
# DARK TRADING DASHBOARD STYLE
# -----------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #00ff9f;
}

div.stButton > button {
    background-color: #1f77ff;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #0057ff;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("trade_probability_model.pkl")
features = joblib.load("model_features.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("# 🤖 TradeAI Quant Analyzer")
st.caption("AI-powered trade setup probability engine")

# Optional logo
try:
    st.image("ai_logo.png", width=120)
except:
    pass

st.write(
"""
Evaluate trading setups using **market structure, liquidity sweeps,
and volatility conditions** to estimate the probability of trade success.
"""
)

# -----------------------------
# TWO PANEL DASHBOARD LAYOUT
# -----------------------------
col1, col2 = st.columns([1,1])

# -----------------------------
# LEFT PANEL — INPUTS
# -----------------------------
with col1:

    st.subheader("Market Conditions")

    trend = st.selectbox(
        "Trend Context",
        ["Bullish", "Bearish"]
    )

    bos = st.selectbox(
        "Break of Structure Direction",
        ["Long", "Short"]
    )

    liquidity = st.checkbox("Liquidity Sweep")

    keylevel = st.selectbox(
        "Key Level Interaction",
        ["Support", "Resistance", "None"]
    )

    volatility = st.selectbox(
        "Market Volatility",
        ["High", "Medium", "Low"]
    )

    session = st.selectbox(
        "Trading Session",
        ["London", "NY", "Asia"]
    )

    oc_break = st.checkbox("Orderflow Break")

    zone = st.selectbox(
        "Premium / Discount Zone",
        ["Premium", "Discount", "Equilibrium"]
    )

    predict_button = st.button("Run AI Trade Evaluation")


# -----------------------------
# RIGHT PANEL — RESULTS
# -----------------------------
with col2:

    st.subheader("AI Trade Evaluation")

    if predict_button:

        data = {
            "trend context": trend,
            "bos direction": bos,
            "liquidity sweep": liquidity,
            "keylevel": keylevel,
            "volatility": volatility,
            "trading session": session,
            "Oc_break": oc_break,
            "Premium_Discount": zone
        }

        df = pd.DataFrame([data])

        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=features, fill_value=0)

        prediction = model.predict(df_encoded)[0]
        probability = model.predict_proba(df_encoded)[0][1]

        prob = probability * 100

        # Prediction
        if prediction == 1:
            st.success("HIGH PROBABILITY TRADE")
        else:
            st.error("LOW PROBABILITY TRADE")

        # -----------------------------
        # PROBABILITY GAUGE
        # -----------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Trade Success Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))

        st.plotly_chart(fig)

        # -----------------------------
        # SIGNAL STRENGTH
        # -----------------------------
        if prob >= 75:
            st.success("STRONG BUY SETUP")
        elif prob >= 55:
            st.warning("MODERATE SETUP")
        else:
            st.error("WEAK SETUP — AVOID TRADE")

        st.write("Model Confidence:", round(prob, 2), "%")

        # -----------------------------
        # HUMAN FRIENDLY EXPLANATIONS
        # -----------------------------
        st.subheader("Key Factors Influencing This Prediction")

        explanations = {
            "trend context_Bullish": "Bullish market trend detected",
            "trend context_Bearish": "Bearish market trend detected",
            "bos direction_Long": "Break of structure in bullish direction",
            "bos direction_Short": "Break of structure in bearish direction",
            "liquidity sweep": "Liquidity sweep detected",
            "keylevel_Support": "Support level interaction",
            "keylevel_Resistance": "Resistance level interaction",
            "volatility_High": "High market volatility",
            "volatility_Low": "Low volatility environment",
            "trading session_London": "London session activity",
            "trading session_NY": "New York session activity",
            "trading session_Asia": "Asian session activity",
            "Premium_Discount_Discount": "Discount entry zone",
            "Premium_Discount_Premium": "Premium entry zone"
        }

        importances = model.feature_importances_

        top_features = sorted(
            zip(features, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for feature, importance in top_features:

            readable = explanations.get(feature, feature)

            st.write(
                f"{readable} — {round(importance * 100,2)}% influence"
            )
