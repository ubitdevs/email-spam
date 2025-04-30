import requests
from typing import Union
import streamlit as st
import re
import numpy as np
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

st.set_page_config(
    page_title="SpamShield - Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

page_bg = """
<style>
body {
background: linear-gradient(to right, #6a11cb, #2575fc);
color: white;
}
h1, h2, h3, h4, h5 {
color: white;
}
.stTextInput>div>div>input, .stTextArea>div>textarea {
background-color: #f0f2f6;
color: #000000;
}
.stButton>button {
background-color: #ff4b4b;
color: white;
border: None;
padding: 0.6em 1.2em;
font-size: 1.1em;
border-radius: 8px;
}
.stButton>button:hover {
background-color: #ff3333;
}
.stMarkdown, .stSubheader {
color: black;
}
.stContainer {
background-color: rgba(255, 255, 255, 0.05);
border-radius: 20px;
padding: 20px;
margin: 10px 0;
}
footer {
visibility: hidden;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: -40px; margin-left:20px;'>
    <img src='https://cdn-icons-png.flaticon.com/512/561/561127.png' width='120' style='margin-bottom: 3px;margin-left: -20px'>
    <h1 style='font-size: 3em;'>SpamShield 🚀</h1>
    <h3 style='margin-top: -40px;'>Your Ultimate Email Spam Detector</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

@st.cache_resource
def download_nltk_data():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt_tab")

download_nltk_data()

@st.cache_resource
def load_models():
    word2vec_model = Word2Vec.load("./model/hamvsspamword2vec.model")
    classifier_model = joblib.load("./model/hamvsspamclassifier.pkl")
    return word2vec_model, classifier_model

word2vecmodel, classifier = load_models()

def clean_text(text):
    text = re.sub(r"(?<=\d)\.(?=\d)", "", text)
    text = re.sub(r"(?<=\w)\\'(?=\w)", "", text)
    text = re.sub(r"(?<=\w)\'(?=\w)", "", text)
    text = re.sub(r"&lt;#&gt;", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

def average_word2vec(doc):
    vectors = [word2vecmodel.wv[word] for word in doc if word in word2vecmodel.wv.index_to_key]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vecmodel.vector_size)

def predict(text):
    cleaned_text = clean_text(text)
    preprocessed_text = preprocess_text(cleaned_text)
    tokenized_text = word_tokenize(preprocessed_text)
    mean_vector = average_word2vec(tokenized_text)
    return classifier.predict([mean_vector])[0]

class Validate:
    def __init__(self, key: str, format: str = "json") -> None:
        self.key = key
        self.format = format
        self.base_url = f"https://www.ipqualityscore.com/api/{self.format}/"

    def email_validation_api(self, email: str, timeout: int = 7, fast: str = 'false', abuse_strictness: int = 0) -> dict:
        url = f"{self.base_url}email/{self.key}/{email}"
        params = {"timeout": timeout, "fast": fast, "abuse_strictness": abuse_strictness}
        response = requests.get(url, params=params)
        return response.json()

API_KEY = os.environ.get("IP_QUALITY_API_KEY")
validate = Validate(API_KEY)

def check_domain_spam(domain: str) -> bool:
    try:
        response = validate.email_validation_api(domain)
        return response.get('fraud_score')
    except Exception:
        return False


if "history" not in st.session_state:
    st.session_state.history = []


with st.container():
    st.subheader("📨 Analyze a New Email")
    with st.form(key="spam_form"):
        sender_email = st.text_input("📬 Sender Email Address", placeholder="someone@example.com")
        message = st.text_area("✉️ Email Message", placeholder="Type your email content here...", height=200)
        submitted = st.form_submit_button("🔍 Scan for Spam")

if submitted:
    if not message.strip():
        st.warning("⚠️ Please enter a message first.")
    else:
        prediction_local = predict(message)
        domain_spam = False

        if sender_email and "@" in sender_email:
            domain = sender_email.split("@")[1]
            domain_spam = check_domain_spam(domain)

        final_decision = "not spam"
        if prediction_local == 1 or domain_spam > 0:
            final_decision = "spam"

        if final_decision == "spam":
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")

        st.session_state.history.append({
            "message": message,
            "sender": sender_email if sender_email else "Unknown",
            "result": final_decision.upper()
        })

st.markdown("---")

with st.container():
    st.subheader("📜 Prediction History")
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            is_spam = entry['result'] == "SPAM"
            sender_color = "#FF4B4B" if is_spam else "#2ECC71"  # Red for spam, Green for not spam
            background_color = "#F5F5F5"  # Light grey for container background
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="background-color: {background_color}; padding: 15px; border-radius: 10px;">
                        <div style="margin-bottom: 10px;">
                            <span style="background-color: {sender_color}; color: white; padding: 5px 10px; border-radius: 20px; font-size: 14px;">
                                {entry['sender']}
                            </span>
                        </div>
                        <div style="font-size: 16px; margin-bottom: 10px;">
                            <strong>Message:</strong> {entry['message']}
                        </div>
                        <div style="font-size: 16px;">
                            <strong>Result:</strong> <span style="color: {sender_color}; font-weight: bold;">📝 {entry['result']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("🕵️‍♂️ No messages analyzed yet.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size: 0.9em;'>"
    "Made with ❤️ by <b>Raza Jaun & Group</b> 🚀"
    "</div>",
    unsafe_allow_html=True
)
