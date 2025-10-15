# ============================================================
# 📘 Chat with PDF — An Intelligent RAG App for Document Insight
# ============================================================
# 🧠 Author: Lavanya Srivastava
# 📍 Country: India | 🌐 Global AI Developer & Trainer
# 🗓️ Year: 2025
# 🔗 Live Demo: https://w7hmkx6rpbmvp6obpnkcby.streamlit.app/
#
# ⚖️ Copyright (c) 2025 Lavanya Srivastava. All Rights Reserved.
# ------------------------------------------------------------
# This project is proprietary intellectual property.
# Unauthorized reproduction, redistribution, or modification,
# in any form or medium, without explicit written permission
# from the author is strictly prohibited.
#
# For educational or research citation:
# “Chat with PDF — Developed by Lavanya Srivastava (2025).”
#
# This application demonstrates production-grade skills in:
#  • LangChain RAG pipeline design
#  • FAISS vector indexing
#  • Streamlit-based conversational UX
#  • Secure key management using st.secrets and .env fallback
#  • Cloud deployment (Streamlit / Azure / Local hybrid)
# ============================================================

import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Free local embeddings + FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ───────────────────────── Setup ─────────────────────────
load_dotenv()

CHAT_BACKEND = os.getenv("CHAT_BACKEND", "groq").lower().strip()   # "groq" or "gemini"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

INDEX_DIR = "faiss_index"

# Free, local embeddings (no quotas)
EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ──────────────────────── Helpers ────────────────────────
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs or []:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=EMBEDDINGS)
    vector_store.save_local(INDEX_DIR)

def get_conversational_chain():
    """Return a QA chain using either Groq (Llama) or Gemini, based on env."""
    prompt_template = """
    Answer the question as thoroughly as possible from the provided context.
    If the answer is not present in the context, say:
    "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    if CHAT_BACKEND == "groq":
        try:
            from langchain_groq import ChatGroq
        except Exception as e:
            st.error(f"Groq backend selected but langchain_groq not installed: {e}")
            raise

        if not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found in .env. Set it or switch CHAT_BACKEND to 'gemini'.")
            raise RuntimeError("Missing GROQ_API_KEY")

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=1024,
        )
        return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    try:
        import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as e:
        st.error(f"Gemini backend selected but Google packages missing: {e}")
        raise

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found in .env. Set it or switch CHAT_BACKEND to 'groq'.")
        raise RuntimeError("Missing GOOGLE_API_KEY")

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        raise

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_version="v1",
        temperature=0.3,
        max_output_tokens=1024,
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def user_input(user_question: str):
    new_db = FAISS.load_local(
        INDEX_DIR, EMBEDDINGS, allow_dangerous_deserialization=True
    )
    docs = new_db.similarity_search(user_question, k=4)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    st.write("Reply:", response["output_text"])

# ───────────────────────── UI ─────────────────────────
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF 💁")

    st.caption(f"Backend: **{CHAT_BACKEND.upper()}**  |  Embeddings: **all-MiniLM-L6-v2**  |  Index: **{INDEX_DIR}**")

    user_question = st.text_input("Ask a question about your PDFs")
    if user_question:
        if not os.path.isdir(INDEX_DIR):
            st.error("No index found. Please upload PDFs and click 'Submit & Process' first.")
        else:
            try:
                user_input(user_question)
            except Exception as e:
                st.error(f"Error answering: {e}")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload PDFs, then click 'Submit & Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing…"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        chunks = get_text_chunks(raw_text)
                        if os.path.isdir(INDEX_DIR):
                            shutil.rmtree(INDEX_DIR)
                        get_vector_store(chunks)
                        st.success("Done! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Failed to process PDFs: {e}")

if __name__ == "__main__":
    main()
