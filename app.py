import streamlit as st
from src.inference import load_artifacts, predict

st.title("News Classifier")

@st.cache_resource
def load_model():
    return load_artifacts()

model, tokenizer, label_encoder, device = load_model()

text = st.text_area("Enter news headline or description")

if st.button("Predict"):
    if text.strip():
        category = predict(text, model, tokenizer, label_encoder, device)
        st.success(f"Category: **{category}**")
    else:
        st.warning("Enter some text first.")