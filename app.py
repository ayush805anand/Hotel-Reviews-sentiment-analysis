# AYUSH ANAND
# Hotel Review Sentiment Analysis Web App

import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

st.set_page_config(
    page_title="Hotel Review Sentiment Analyzer",
    layout="centered")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("picture.jpg", width=300)
st.title("Hotel Review Sentiment Analysis")
st.write("Enter a hotel review to predict its sentiment.")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
MAX_LEN = 200

@st.cache_resource
def load_artifacts():
    model = load_model("bilstm_sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder
model, tokenizer, label_encoder = load_artifacts()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)
def color_sentiment(sentiment):
    if sentiment.lower() == "positive":
        return "green"
    elif sentiment.lower() == "neutral":
        return "orange"
    else:
        return "red"
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    sorted_idx = np.argsort(probs)[::-1]
    top_idx = sorted_idx[0]
    second_idx = sorted_idx[1]
    top_sentiment = label_encoder.inverse_transform([top_idx])[0].capitalize()
    second_sentiment = label_encoder.inverse_transform([second_idx])[0].capitalize()
    top_conf = probs[top_idx] * 100
    second_conf = probs[second_idx] * 100
    return top_sentiment, top_conf, second_sentiment, second_conf

user_input = st.text_area(
    "Hotel Review Text",
    height=150,
    placeholder="Type or paste a hotel review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        top_sentiment, top_conf, second_sentiment, second_conf = predict_sentiment(user_input)
        color = color_sentiment(top_sentiment)
        st.subheader("Result")
        st.markdown(
            f"""
            <b>Predicted Sentiment:</b>
            <span style="color:{color}; font-weight:bold;">
                {top_sentiment}
            </span>
            """,
            unsafe_allow_html=True)
        st.markdown(
            f"""
            <b>Confidence:</b>
            <span style="color:{color}; font-weight:bold;">
                {top_conf:.2f}%
            </span>
            """,
            unsafe_allow_html=True)
        st.markdown(
            f"**Second Most Likely:** {second_sentiment} ({second_conf:.2f}%)")

st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Made by Ayush Anand - Final Capstone Project 3 - IITG Course</p>",
    unsafe_allow_html=True)