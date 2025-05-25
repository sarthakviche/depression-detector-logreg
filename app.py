import streamlit as st
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load model components
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
best_threshold = joblib.load("best_threshold.pkl")

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# App title
st.title("Depression Tweet Classifier")

# Text input
tweet = st.text_area("Enter a tweet to analyze:")

if st.button("Classify"):
    if not tweet.strip():
        st.warning("Please enter a tweet.")
    else:
        # Calculate sentiment using VADER
        sentiment_score = analyzer.polarity_scores(tweet)["compound"]
        st.write(f"Auto-detected Sentiment Score: `{sentiment_score:.2f}`")

        # Transform tweet text to TF-IDF
        X_tfidf = vectorizer.transform([tweet])
        X_input = np.hstack([X_tfidf.toarray(), [[sentiment_score]]])  # add sentiment

        # Predict
        prob = model.predict_proba(X_input)[:, 1][0]
        label = int(prob >= best_threshold)

        st.write(f"Prediction Probability: `{prob:.2f}`")
        if label == 1:
            st.error("⚠️ This tweet is **depressive**.")
        else:
            st.success("✅ This tweet is **not depressive**.")
