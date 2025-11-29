"""
PDF Answer Generator (Local QA + Streamlit)
-------------------------------------------

- No API keys, no cloud calls
- Uses local Transformer QA model to extract exact answers from PDF
- Attractive Streamlit web UI
- Steps:
    1. Upload PDF
    2. Ask question
    3. Get exact answer + score + page + context snippet

Run:
    streamlit run pdf_answer_streamlit_qa.py
"""

import io
import textwrap

import PyPDF2
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# ============ PDF & Text Utilities ============

def extract_pdf_pages(file_bytes):
    """Return list of (page_number, text) from PDF bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((i + 1, text))
    return pages


def chunk_pages(pages, max_chars=800, overlap=150):
    """
    Split each page text into overlapping chunks to keep context small enough
    for the QA model.
    Returns list of dicts: {page, text, chunk_id}
    """
    chunks = []
    chunk_id = 0
    for page_num, text in pages:
        t = text.replace("\r", " ").replace("\n", " ")
        t = " ".join(t.split())  # normalize spaces
        if not t:
            continue

        start = 0
        while start < len(t):
            end = min(len(t), start + max_chars)
            chunk_text = t[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_num,
                        "text": chunk_text,
                    }
                )
                chunk_id += 1
            start += max_chars - overlap

    return chunks


def build_tfidf_index(chunks):
    """Build TF-IDF vectorizer + matrix from chunk texts."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def get_top_chunks(question, chunks, vectorizer, matrix, top_k=5):
    """Retrieve top_k most relevant chunks by TF-IDF cosine similarity."""
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]
    idxs = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in idxs:
        c = chunks[idx]
        results.append(
            {
                "page": c["page"],
                "text": c["text"],
                "score": float(scores[idx]),
            }
        )
    return results


# ============ QA Model (Local) ============

@st.cache_resource
def load_qa_model():
    """
    Load a small, fast QA model.
    This is cached so it loads only once.
    """
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return qa


def answer_from_chunks(question, candidate_chunks, qa_model):
    """
    Run the QA model on the top candidate chunks to get the best answer.
    Returns best answer dict or None.
    """
    best = None
    for chunk in candidate_chunks:
        context = chunk["text"]
        if not context.strip():
            continue
        try:
            out = qa_model(question=question, context=context)
        except Exception:
            continue

        answer_text = out.get("answer", "").strip()
        score = float(out.get("score", 0.0))

        if not answer_text:
            continue

        result = {
            "answer": answer_text,
            "answer_score": score,
            "page": chunk["page"],
            "chunk_score": chunk["score"],
            "context": context,
        }

        if best is None or score > best["answer_score"]:
            best = result

    return best


# ============ STREAMLIT UI ============

st.set_page_config(
    page_title="PDF Answer Generator (Local QA)",
    page_icon="📄",
    layout="wide",
)

# --- Custom styling ---
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #020617 60%, #1f2937 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .block-container {
        padding-top: 2rem;
    }
    .card {
        background: #020617;
        border-radius: 1rem;
        padding: 1rem 1.25rem;
        border: 1px solid #1f2937;
        box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    }
    .title-text {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg,#38bdf8,#a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle-text {
        color: #9ca3af;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
      <div class="title-text">📄 Local PDF Answer Generator</div>
      <p class="subtitle-text">
        Upload a PDF, ask a question, and this app will extract the most likely answer using a local AI model.
        No API keys. No internet. Everything happens on your machine.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

col_left, col_right = st.columns([2, 3], gap="large")

with col_left:
    st.markdown("### 1️⃣ Upload your PDF")
    uploaded = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="The text will be extracted locally from this file.",
    )

    st.markdown("### 2️⃣ Ask your question")
    question = st.text_input(
        "Question",
        placeholder="e.g. What is the main objective of this report?",
    )

    top_k = st.slider(
        "Search depth (number of text chunks to consider)",
        min_value=3,
        max_value=15,
        value=7,
        help="Higher = more thorough but slower.",
    )

    run_btn = st.button("🔍 Get Answer", use_container_width=True)

with col_right:
    st.markdown("### 🧠 Answer & Evidence")
    answer_area = st.empty()

# Process PDF & QA
if uploaded is not None:
    bytes_data = uploaded.read()

    with st.spinner("📑 Extracting pages..."):
        pages = extract_pdf_pages(bytes_data)
        pages = [(p, t) for (p, t) in pages if t.strip()]

    if not pages:
        st.error("No extractable text found in this PDF. It may be a scanned image-only document.")
    else:
        with st.spinner("🔎 Chunking pages and building search index..."):
            chunks = chunk_pages(pages, max_chars=600, overlap=120)
            vectorizer, matrix = build_tfidf_index(chunks)

        st.success(f"Indexed {len(chunks)} text chunks from {len(pages)} pages.")

        if run_btn and question.strip():
            with st.spinner("🤖 Thinking... (local QA model running)"):
                qa_model = load_qa_model()
                candidate_chunks = get_top_chunks(question, chunks, vectorizer, matrix, top_k=top_k)
                best = answer_from_chunks(question, candidate_chunks, qa_model)

            if best is None:
                answer_area.warning("I couldn't confidently find an answer in this document. Try rephrasing your question.")
            else:
                ans_text = best["answer"]
                ans_score = best["answer_score"]
                ans_page = best["page"]
                ctx = best["context"]

                with answer_area.container():
                    st.markdown(
                        f"""
                        <div class="card">
                          <h3 style="color:#e5e7eb;">✅ Extracted Answer</h3>
                          <p style="font-size:1.1rem; color:#f9fafb;"><b>{ans_text}</b></p>
                          <p style="color:#9ca3af; font-size:0.85rem;">
                            Confidence: <code>{ans_score:.3f}</code> &nbsp;|&nbsp; Page: <b>{ans_page}</b>
                          </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### 🔍 Supporting context")
                    st.markdown(
                        f"<div style='background:#020617;border-radius:0.75rem;padding:0.75rem 1rem;border:1px solid #111827;'>"
                        f"<span style='color:#9ca3af;font-size:0.85rem;'>Excerpt from page {ans_page}:</span><br><br>"
                        f"<span style='color:#e5e7eb;font-size:0.95rem;'>{textwrap.fill(ctx, width=120).replace(chr(10), '<br>')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    with st.expander("Show other candidate chunks"):
                        for i, c in enumerate(candidate_chunks, start=1):
                            st.markdown(
                                f"**[{i}] Page {c['page']} — Search score: `{c['score']:.3f}`**"
                            )
                            st.write(textwrap.fill(c["text"], width=120))
                            st.markdown("---")

        elif run_btn and not question.strip():
            st.warning("Please type a question first.")
else:
    st.info("👆 Upload a PDF on the left to get started.")
