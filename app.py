import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
INDEX_PATH = "faiss_index"

st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄")

if not GOOGLE_API_KEY:
    st.warning("Google API key is missing. Enter it below to continue.")
    GOOGLE_API_KEY = st.text_input(
        "Google API key",
        type="password",
        help="Add it to your deployment environment as GOOGLE_API_KEY or enter it here for this session.",
    )
    if GOOGLE_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    else:
        st.stop()


def read_pdfs(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text() or ""
            if text:
                all_text += text + "\n"
    return all_text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vector_store(chunks, index_path=INDEX_PATH):
    embedding = create_embeddings()
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embedding)
    vector_store.save_local(index_path)
    return vector_store


def load_vector_store(index_path=INDEX_PATH):
    embeddings = create_embeddings()
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_chunks(question, k=5, index_path=INDEX_PATH):
    vector_store = load_vector_store(index_path)
    return vector_store.similarity_search(question, k=k)


def get_prompt_and_llm():
    prompt_template = """
You are an AI assistant.
Answer the question using only the context below.
If the answer is not present in the context, say exactly:
"THE ANSWER IS NOT AVAILABLE IN THE PROVIDED CONTEXT."
The answer has to be in bullet point wise, each point not exceeding 200 words.
The answer has to be in a way that a CLASS 10TH GRADE STUDENT UNDERSTANDS IT.

Context:
{context}
Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)
    return prompt, llm


def answer_question(question):
    docs = retrieve_chunks(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt, llm = get_prompt_and_llm()
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    return response.content


def build_index_from_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    text = read_pdfs([temp_pdf_path])
    chunks = split_text(text)
    if not chunks:
        return None

    build_vector_store(chunks)
    return len(chunks)


st.set_page_config(page_title="PDF Chat Assistant", page_icon="📄")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f3f7ff 0%, #eef9f5 100%);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        h1 {
            font-weight: 800;
            color: #0f172a;
            letter-spacing: -0.04em;
            margin-bottom: 0.3rem;
        }

        .subheader {
            color: #475569;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(6px);
        }

        .stFileUploader > div {
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 16px;
        }

        div[data-testid="stButton"] > button {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.65rem 1.2rem;
            box-shadow: 0 10px 24px rgba(79, 70, 229, 0.28);
        }

        div[data-testid="stButton"] > button:hover {
            filter: brightness(1.02);
            transform: translateY(-1px);
        }

        .stTextInput > div > div > input {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.5);
            padding: 0.8rem 0.9rem;
            background: rgba(255,255,255,0.9);
        }

        .stAlert {
            border-radius: 12px;
            border: none;
        }

        .answer-box {
            background: linear-gradient(135deg, #f8fafc 0%, #f0fdf4 100%);
            border: 1px solid rgba(34, 197, 94, 0.2);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            color: #0f172a;
            line-height: 1.7;
            white-space: pre-wrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 PDF Chat Assistant")
st.markdown('<div class="subheader">Upload a PDF and ask questions about it in plain English.</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("Build PDF Index"):
            with st.spinner("Reading and indexing the PDF..."):
                chunk_count = build_index_from_uploaded_pdf(uploaded_file)
            if chunk_count:
                st.success(f"Index built successfully using {chunk_count} chunks.")
            else:
                st.warning("No text could be extracted from the PDF.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    question = st.text_input("Ask a question about the PDF", placeholder="Example: What is multi-head attention?")

    if st.button("Get Answer") and question:
        try:
            with st.spinner("Searching the PDF and generating the answer..."):
                answer = answer_question(question)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
