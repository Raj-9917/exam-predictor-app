import streamlit as st
import pandas as pd
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io
import openai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

# === OCR and PDF Parsing ===
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

# === GPT-based Question Cleaner ===
def clean_questions_with_gpt(raw_text):
    prompt = f"""
You are a smart assistant for students. The following text is from a scanned exam paper. It may contain college headers, subject codes, instructions, or actual questions.

Your task:
1. Remove all lines like 'Subject Code', 'University Name', 'Time Allowed', 'Marks', 'Attempt any four', 'Note', etc.
2. Keep only the actual questions (with or without numbering).
3. Return the cleaned list as separate lines.

Text:
{raw_text}

Cleaned Questions:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract only the questions from a messy scanned exam file."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        output = response["choices"][0]["message"]["content"]
        return [line.strip() for line in output.strip().split("\n") if len(line.strip()) > 10]
    except Exception as e:
        st.error(f"GPT cleanup failed: {e}")
        return []

# === GPT Topic Detector ===
def detect_topic_gpt(question):
    prompt = f"""
You are an intelligent assistant. Given the question below, identify the most relevant topic (e.g., Stack, Queue, Tree, Graph, Sorting, Recursion, etc.). Reply with only ONE topic name.

Question: \"{question}\"
Topic:"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for exam preparation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=10
        )
        topic = response["choices"][0]["message"]["content"].strip()
        return topic
    except Exception as e:
        print("GPT Error:", e)
        return "Unknown"

# === Streamlit App ===
st.set_page_config(page_title="GPT Exam Cleaner & Predictor", layout="centered")
st.title("📚 GPT-Based Exam Question Extractor")

uploaded_file = st.file_uploader("📄 Upload a Question Paper (PDF/Image)", type=["pdf", "jpg", "png"])
manual_input = st.text_area("✍️ Or paste questions manually (raw scan text):", height=200)

questions = []
extracted = ""

if uploaded_file:
    if "pdf" in uploaded_file.type:
        extracted = extract_text_from_pdf(uploaded_file)
    else:
        extracted = extract_text_from_image(uploaded_file.read())
    questions = clean_questions_with_gpt(extracted)
elif manual_input:
    extracted = manual_input
    questions = clean_questions_with_gpt(manual_input)

if questions:
    st.success(f"✅ {len(questions)} questions extracted successfully using GPT.")

    data = []
    for q in questions:
        topic = detect_topic_gpt(q)
        data.append((q, topic))

    df = pd.DataFrame(data, columns=["Question", "Topic"])
    topic_counts = df["Topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Frequency"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Question"])
    similarity = cosine_similarity(X, X)
    df["Prediction Score"] = pd.DataFrame(similarity).sum(axis=1)
    df_sorted = df.sort_values(by="Prediction Score", ascending=False)

    with st.expander("📄 Original Extracted Text"):
        st.text(extracted)

    with st.expander("✅ GPT Cleaned Questions"):
        st.write(questions)

    st.subheader("🔮 Top Predicted Questions")
    st.dataframe(df_sorted.head(10), use_container_width=True)

    st.subheader("📊 Most Repeated Topics")
    st.dataframe(topic_counts, use_container_width=True)

    st.subheader("📘 All Questions with Topics")
    with st.expander("📖 Show Full Table"):
        st.dataframe(df_sorted, use_container_width=True)
else:
    st.info("Upload or paste scanned text to extract real questions using GPT.")

st.markdown("---")
st.caption("🤖 Built by Shashank Verma • GPT Exam Assistant")
