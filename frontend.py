import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model and feature names
model = joblib.load("trade_probability_model.pkl")
features = joblib.load("model_features.pkl")

st.title("AI Trade Probability Predictor")

st.write("Enter market conditions to evaluate trade probability")

trend = st.selectbox("Trend Context", ["Bullish", "Bearish"])
bos = st.selectbox("BOS Direction", ["Long", "Short"])
liquidity = st.checkbox("Liquidity Sweep")
keylevel = st.selectbox("Key Level", ["Support", "Resistance", "None"])
volatility = st.selectbox("Volatility", ["High", "Medium", "Low"])
session = st.selectbox("Trading Session", ["London", "NY", "Asia"])
oc_break = st.checkbox("Orderflow Break")
zone = st.selectbox("Premium / Discount", ["Premium", "Discount", "Equilibrium"])


if st.button("Predict Trade"):

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

    if prediction == 1:
        st.success("High Probability Trade")
    else:
        st.error("Low Probability Trade")

    # Gauge chart
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

    # Signal strength
    if prob >= 75:
        st.success("STRONG BUY SETUP")
    elif prob >= 55:
        st.warning("MODERATE SETUP")
    else:
        st.error("WEAK SETUP - AVOID TRADE")

    st.write("Model Confidence:", round(prob, 2), "%")

    # Human readable explanations
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
        "trading session_London": "London trading session activity",
        "trading session_NY": "New York trading session activity",
        "trading session_Asia": "Asian trading session",
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
        st.write(f"{readable} — {round(importance*100,2)}% influence")
