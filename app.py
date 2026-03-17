"""
Streamlit Mini PDF-Q&A – Groq Edition with chat history context
"""
import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: #111317;
        color: #f1f1f1;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    .glass {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(7px);
        -webkit-backdrop-filter: blur(7px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .gradient-text {
        font-weight: 800;
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #ff8c00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .neon-metric {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 12px;
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid rgba(0, 255, 255, 0.4);
        color: #00ffff;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .user-bubble {
        background: rgba(66, 133, 244, 0.25);
        border-radius: 18px 18px 0 18px;
        padding: 0.8rem 1.2rem;
        width: fit-content;
        max-width: 70%;
        margin-left: auto;
        margin-bottom: 1rem;
        border: 1px solid rgba(66, 133, 244, 0.5);
    }
    .bot-bubble {
        background: rgba(255, 75, 75, 0.25);
        border-radius: 18px 18px 18px 0;
        padding: 0.8rem 1.2rem;
        width: fit-content;
        max-width: 70%;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 75, 75, 0.5);
    }
    .stFileUploader > div {
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.3);
    }
    .stFileUploader > div:hover {
        border-color: #ff4b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- HEADER ----------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<p class="gradient-text" style="text-align:center;">PDF AI Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; opacity:0.7;">Upload a PDF → ask questions → get instant answers from the document</p>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- AUTH ----------
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("🔑 Please set GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# ---------- UTILS ----------
@st.cache_data(show_spinner=False)
def parse_pdf(file):
    reader = PdfReader(file)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text, len(reader.pages)

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    vs = FAISS.from_texts(chunks, get_embeddings())
    return vs, len(chunks)

# ---------- SESSION ----------
for k in ["messages", "vs", "stats"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k == "messages" else None if k == "vs" else {}

# ---------- UPLOADER ----------
uploaded = st.file_uploader(
    " ",
    type=["pdf"],
    help="Drag & drop or click to select a PDF (max 200 MB)",
)

if uploaded and st.session_state.vs is None:
    with st.spinner("Parsing & embedding…"):
        raw_text, pages = parse_pdf(uploaded)
        vs, chunks = build_vectorstore(raw_text)
        st.session_state.vs = vs
        st.session_state.stats = {"pages": pages, "chunks": chunks}
    st.success("✅ PDF indexed successfully!")

# ---------- METRICS ----------
if st.session_state.vs:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<span class="neon-metric">Pages: {st.session_state.stats["pages"]}</span>', unsafe_allow_html=True)
    c2.markdown(f'<span class="neon-metric">Chunks: {st.session_state.stats["chunks"]}</span>', unsafe_allow_html=True)
    if c3.button("🗑️  Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- CHAT HISTORY ----------
st.markdown("---")
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------- INPUT ----------
if prompt := st.chat_input("Ask a question about the PDF"):
    if st.session_state.vs is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # User bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

    # --- Smart retrieval: enrich vague follow-ups with last bot answer ---
    last_bot = next(
        (m["content"] for m in reversed(st.session_state.messages[:-1]) if m["role"] == "assistant"),
        ""
    )
    enriched_query = f"{last_bot} {prompt}" if last_bot else prompt
    docs = st.session_state.vs.similarity_search(enriched_query, k=5)
    context = "\n\n".join(d.page_content for d in docs)

    # System prompt
    system = (
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{context}"
    )

    # Build full message history for Groq
    groq_messages = [{"role": "system", "content": system}]
    for m in st.session_state.messages[:-1]:
        groq_messages.append({"role": m["role"], "content": m["content"]})
    groq_messages.append({"role": "user", "content": prompt})

    # Typing indicator
    with st.empty():
        st.markdown('<div class="bot-bubble">▌</div>', unsafe_allow_html=True)
        time.sleep(0.3)

    # Groq call
    client = Groq(api_key=groq_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        max_tokens=1024,
        temperature=0
    )
    full = response.choices[0].message.content

    st.markdown(f'<div class="bot-bubble">{full}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": full})
