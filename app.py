import streamlit as st
import joblib
import numpy as np

# Load saved model, vectorizer, threshold
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
best_threshold = joblib.load("best_threshold.pkl")

# App title
st.title("Depression Tweet Classifier")

# User input
tweet = st.text_area("Enter a tweet to analyze:")

# Sentiment input (optional, simulate with slider or dropdown if not predicted)
sentiment = st.slider("Sentiment Score (-1 = Negative, 1 = Positive)", -1.0, 1.0, 0.0, step=0.1)

if st.button("Classify"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        # TF-IDF transformation
        X_tfidf = vectorizer.transform([tweet])
        X_input = np.hstack([X_tfidf.toarray(), [[sentiment]]])  # combine with sentiment

        # Predict probability and classify
        prob = model.predict_proba(X_input)[:, 1][0]
        label = int(prob >= best_threshold)

        st.write(f"Prediction Probability: {prob:.2f}")
        st.success("This tweet is **depressive**." if label == 1 else "This tweet is **not depressive**.")
