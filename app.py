from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("trade_probability_model.pkl")

# Load feature columns
features = joblib.load("model_features.pkl")


@app.get("/")
def home():
    return {"message": "Trade AI Backend Running"}


@app.post("/predict")
def predict_trade(trade: dict):

    df = pd.DataFrame([trade])

    df_encoded = pd.get_dummies(df)

    df_encoded = df_encoded.reindex(columns=features, fill_value=0)

    prediction = model.predict(df_encoded)[0]

    probability = model.predict_proba(df_encoded)[0][1]

    # Calculate feature importance
    importances = model.feature_importances_

    top_features = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "top_features": top_features
    }