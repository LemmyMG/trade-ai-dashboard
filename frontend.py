import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load trained model and features
model = joblib.load("trade_probability_model.pkl")
features = joblib.load("model_features.pkl")

st.title("AI Trade Probability Predictor")

st.write("Enter market conditions to evaluate trade probability")

# Inputs
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

    # Convert to dataframe
    df = pd.DataFrame([data])

    # Encode categorical variables
    df_encoded = pd.get_dummies(df)

    # Match training feature structure
    df_encoded = df_encoded.reindex(columns=features, fill_value=0)

    # Make prediction
    prediction = model.predict(df_encoded)[0]
    probability = model.predict_proba(df_encoded)[0][1]

    # Display prediction
    if prediction == 1:
        st.success("High Probability Trade")
    else:
        st.error("Low Probability Trade")

    prob = probability * 100

    # Gauge meter
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
