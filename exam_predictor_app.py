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

# === GPT Topic Detector ===
def detect_topic_gpt(question):
    prompt = f"""You are an intelligent assistant. Given the question below, identify the most relevant topic (e.g., Stack, Queue, Tree, Graph, Sorting, Recursion, etc.). Reply with only ONE topic name.

Question: "{question}"
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

# === GPT Explanation Generator (optional) ===
def explain_question_gpt(question):
    prompt = f"Explain this question briefly in simple terms for students:\n\n{question}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=120
        )
        return response["choices"][0]["message"]["content"].strip()
    except:
        return "Explanation not available."

# === Streamlit App UI ===
st.set_page_config(page_title="GPT-Based Exam Predictor", layout="centered")
st.title("📚 AI Exam Question & Topic Predictor")
st.markdown("Upload a question paper or type questions manually. App uses GPT to predict topics and most repeated questions.")

uploaded_file = st.file_uploader("📄 Upload a PDF/Image", type=["pdf", "jpg", "png"])
manual_input = st.text_area("✍️ Or paste questions manually (one per line):", height=200)

questions = []

if uploaded_file:
    if "pdf" in uploaded_file.type:
        extracted = extract_text_from_pdf(uploaded_file)
    else:
        extracted = extract_text_from_image(uploaded_file.read())
    questions = [line.strip() for line in extracted.split("\n") if line.strip()]
elif manual_input:
    questions = [line.strip() for line in manual_input.split("\n") if line.strip()]

# === Analyze Questions ===
if questions:
    st.success(f"{len(questions)} questions loaded. Generating topics using GPT...")

    data = []
    for q in questions:
        topic = detect_topic_gpt(q)
        data.append((q, topic))

    df = pd.DataFrame(data, columns=["Question", "Topic"])
    
    # Frequency of each topic
    topic_counts = df["Topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Frequency"]

    # Similarity-based prediction score
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Question"])
    similarity = cosine_similarity(X, X)
    df["Prediction Score"] = pd.DataFrame(similarity).sum(axis=1)

    df_sorted = df.sort_values(by="Prediction Score", ascending=False)

    st.subheader("🔮 Top Predicted Questions")a
    st.dataframe(df_sorted.head(10), use_container_width=True)

    st.subheader("📊 Most Repeated Topics")
    st.dataframe(topic_counts, use_container_width=True)

    st.subheader("📘 All Questions with Topics")
    with st.expander("📖 Show Full Table"):
        st.dataframe(df_sorted, use_container_width=True)

    # Optional: Explanation
    st.subheader("💡 Want AI to explain a question?")
    selected_question = st.selectbox("Select a question:", df_sorted["Question"].tolist())
    if st.button("Explain with GPT"):
        with st.spinner("Thinking..."):
            explanation = explain_question_gpt(selected_question)
        st.info(explanation)
else:
    st.info("Upload or type questions to begin.")

st.markdown("---")
st.caption("🚀 Built by Shashank Verma • GPT-Powered Exam Assistant")
