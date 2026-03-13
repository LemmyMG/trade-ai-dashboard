import joblib
import pandas as pd

model = joblib.load("trade_probability_model.pkl")
features = joblib.load("model_features.pkl")

import streamlit as st
import requests
import plotly.graph_objects as go

st.title("AI Trade Probability Predictor")

st.write("Enter market conditions to evaluate trade probability")

# INPUTS
trend = st.selectbox("Trend Context", ["Bullish", "Bearish"])
bos = st.selectbox("BOS Direction", ["Long", "Short"])
liquidity = st.checkbox("Liquidity Sweep")
keylevel = st.selectbox("Key Level", ["Support", "Resistance", "None"])
volatility = st.selectbox("Volatility", ["High", "Medium", "Low"])
session = st.selectbox("Trading Session", ["London", "NY", "Asia"])
oc_break = st.checkbox("Orderflow Break")
zone = st.selectbox("Premium / Discount", ["Premium", "Discount", "Equilibrium"])

# BUTTON
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

result = {
    "prediction": int(prediction),
    "probability": float(probability)
}

    if result["prediction"] == 1:
        st.success("High Probability Trade")
    else:
        st.error("Low Probability Trade")

    prob = result["probability"] * 100

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

    if prob >= 75:
        st.success("STRONG BUY SETUP")
    elif prob >= 55:
        st.warning("MODERATE SETUP")
    else:
        st.error("WEAK SETUP - AVOID TRADE")

    st.subheader("Key Factors Influencing This Prediction")

    for feature, importance in result["top_features"]:

        st.write(f"{feature}: {round(importance * 100, 2)}% influence")
