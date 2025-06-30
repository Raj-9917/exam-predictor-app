# filename: exam_predictor_app.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

# Set tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="Exam Predictor", layout="centered")
st.title("📚 AI Exam Question Predictor")
st.markdown("Upload question papers (PDF/Image) or type manually. The app will predict most likely questions for your exam.")

# Upload Section
st.subheader("Step 1: Upload Past Questions")

uploaded_file = st.file_uploader("Upload a PDF or Image (.pdf, .jpg, .png):", type=["pdf", "jpg", "png"])
manual_input = st.text_area("Or paste questions below (one per line):", height=200)

questions = []

def extract_text_from_pdf(file):
    text = ""
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text

def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text

if uploaded_file:
    file_type = uploaded_file.type
    if "pdf" in file_type:
        extracted = extract_text_from_pdf(uploaded_file)
    elif "image" in file_type or "png" in file_type or "jpeg" in file_type:
        extracted = extract_text_from_image(uploaded_file.read())
    else:
        extracted = ""

    questions = [line.strip() for line in extracted.strip().split("\n") if line.strip()]
    st.success(f"Extracted {len(questions)} lines from uploaded file.")
elif manual_input:
    questions = [line.strip() for line in manual_input.strip().split("\n") if line.strip()]

# Prediction Engine
if questions:
    st.subheader("Step 2: AI Prediction of Important Questions")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions)
    similarity_matrix = cosine_similarity(X, X)
    frequency_score = pd.DataFrame(similarity_matrix).sum(axis=1)

    result_df = pd.DataFrame({
        "Question": questions,
        "Prediction Score": frequency_score
    }).sort_values(by="Prediction Score", ascending=False).reset_index(drop=True)

    st.write("🔮 **Top Predicted Questions:**")
    st.dataframe(result_df.head(10), use_container_width=True)
else:
    st.info("Please upload a question paper or paste questions manually.")

st.markdown("---")
st.caption("Created by Shashank Verma • Powered by AI")
