import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = {
    "text": [
        "I love this product",
        "This is amazing",
        "Absolutely fantastic experience",
        "I hate this",
        "Very bad service",
        "Worst purchase ever",
        "It is okay",
        "Not bad"
    ],
    "sentiment": [
        "positive","positive","positive",
        "negative","negative","negative",
        "neutral","neutral"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

model = LogisticRegression()
model.fit(X, y)

st.title("💬 Sentiment Analyzer App")

user_input = st.text_input("Enter a sentence")

if st.button("Predict"):
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)
    st.success(f"Sentiment: {prediction[0]}")
