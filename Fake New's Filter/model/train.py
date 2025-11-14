# train.py
# -------------------------
# pip install pandas scikit-learn joblib

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("fake_news.csv")

X = df["text"]
y = df["label"].astype(int)

# Create training pipeline
pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(stop_words="english")),
    ("model", MultinomialNB())
])

# Train model
pipeline.fit(X, y)

# Save model + vectorizer
joblib.dump(pipeline, "model.pkl")
joblib.dump(pipeline.named_steps["vectorizer"], "vectorizer.pkl")

print("🎉 Model and vectorizer saved successfully!")
