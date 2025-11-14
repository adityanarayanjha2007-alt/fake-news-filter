import os
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# FastAPI setup

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")


# Load Classifier and Vectorizer

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    pipeline = joblib.load(MODEL_PATH)
    classifier = pipeline.named_steps["model"]
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Classifier and vectorizer loaded successfully.")
else:
    classifier = None
    vectorizer = None
    print("⚠️ Model or vectorizer not found! Please check paths.")


# Input Schema

class NewsInput(BaseModel):
    text: str


# Predict Route

@app.post("/predict")
def predict_news(data: NewsInput):
    if classifier is None or vectorizer is None:
        return {"error": "Classifier or vectorizer not loaded"}

    try:
        text_vectorized = vectorizer.transform([data.text])
        prediction = classifier.predict(text_vectorized)[0]
        probabilities = classifier.predict_proba(text_vectorized)[0]
        probability = float(probabilities[prediction])

        
        label = "Real News ✅" if prediction == 0 else "Fake News ❌"

        return {
            "prediction": str(prediction),
            "label": label,
            "probability": round(probability * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# Health check

@app.get("/")
def home():
    return {"message": "Fake News Detector API is running!"}
