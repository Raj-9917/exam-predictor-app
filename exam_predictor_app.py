import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

# === Topic keywords ===
TOPIC_KEYWORDS = {
    "Stack": ["stack", "push", "pop", "LIFO"],
    "Queue": ["queue", "FIFO", "enqueue", "dequeue"],
    "Tree": ["tree", "binary tree", "BST", "traversal"],
    "Linked List": ["linked list", "singly", "doubly", "node", "pointer"],
    "Recursion": ["recursion", "recursive"],
    "Sorting": ["sort", "bubble", "selection", "insertion", "merge", "quick"],
    "Searching": ["search", "binary search", "linear search"],
    "Hashing": ["hash", "hashing", "collision"],
    "Graph": ["graph", "BFS", "DFS", "adjacency", "vertex", "edge"]
}

# === File extraction functions ===
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

# === Topic Detection ===
def detect_topic(question):
    q = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return topic
    return "Unknown"

# === Streamlit UI ===
st.set_page_config(page_title="AI Exam Predictor", layout="centered")
st.title("📚 AI-Based Exam Question & Topic Predictor")
st.markdown("Upload PDF/image of past year paper or paste questions manually. App detects topics, predicts repeated questions, and shows important topics.")

uploaded_file = st.file_uploader("📄 Upload a question paper (PDF/Image)", type=["pdf", "jpg", "png"])
manual_input = st.text_area("✍️ Or paste questions here (one per line):", height=200)

questions = []

if uploaded_file:
    if "pdf" in uploaded_file.type:
        extracted = extract_text_from_pdf(uploaded_file)
    else:
        extracted = extract_text_from_image(uploaded_file.read())
    questions = [line.strip() for line in extracted.split("\n") if line.strip()]
elif manual_input:
    questions = [line.strip() for line in manual_input.split("\n") if line.strip()]

# === Process & Display ===
if questions:
    st.success(f"{len(questions)} questions loaded.")
    
    # Tag questions with topics
    data = []
    for q in questions:
        topic = detect_topic(q)
        data.append((q, topic))
    
    df = pd.DataFrame(data, columns=["Question", "Topic"])

    # Count topic frequency
    topic_freq = df["Topic"].value_counts().reset_index()
    topic_freq.columns = ["Topic", "Frequency"]

    # Score similarity for question prediction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Question"])
    similarity = cosine_similarity(X, X)
    df["Prediction Score"] = pd.DataFrame(similarity).sum(axis=1)

    df_sorted = df.sort_values(by="Prediction Score", ascending=False)

    st.subheader("🔮 Top Predicted Questions")
    st.dataframe(df_sorted.head(10), use_container_width=True)

    st.subheader("📊 Topic-Wise Question Count")
    st.dataframe(topic_freq, use_container_width=True)
    
    st.subheader("📚 All Detected Questions with Topics")
    st.dataframe(df_sorted, use_container_width=True)
else:
    st.warning("Please upload or type some questions to start prediction.")

st.markdown("---")
st.caption("Built by Shashank Verma • AI-Powered Exam Helper")
