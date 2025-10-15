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
# For citation:
# “Chat with PDF — Developed by Lavanya Srivastava (2025).”
# ============================================================

import os, shutil, streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ---------- Setup ----------
load_dotenv()

# Prefer Streamlit Secrets, then .env
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
os.environ["GROQ_API_KEY"] = GROQ_API_KEY  # make available to SDK

INDEX_DIR = "faiss_index"
EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- Helpers ----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs or []:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=1_000)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=EMBEDDINGS)
    vector_store.save_local(INDEX_DIR)


def get_conversational_chain():
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
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY missing. Add it in Streamlit Secrets.")
        st.stop()

    llm = ChatGroq(model_name="llama-3.1-8b-instant",
                   temperature=0.3,
                   max_tokens=1024)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


def answer_question(user_question: str):
    new_db = FAISS.load_local(INDEX_DIR, EMBEDDINGS,
                              allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=4)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs,
                      "question": user_question},
                     return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# ---------- UI ----------
def main():
    st.set_page_config(
        page_title="Chat PDF",
        page_icon="📄",
        menu_items={
            "About": "Chat with PDF — © 2025 Lavanya Srivastava\n"
                     "Live: https://w7hmkx6rpbmvp6obpnkcby.streamlit.app/\n"
                     "Author: https://www.linkedin.com/in/lavanya-srivastava"
        },
    )

    st.header("Chat with PDF 💁")
    st.markdown(
        "Built by **Lavanya Srivastava** · "
        "[LinkedIn](https://www.linkedin.com/in/lavanya-srivastava) · "
        "[Live App](https://w7hmkx6rpbmvp6obpnkcby.streamlit.app/)"
    )
    st.caption(f"Backend: **GROQ** | Embeddings: **all-MiniLM-L6-v2** | Index: **{INDEX_DIR}**")

    # Footer credit
    st.markdown(
        """
        <style>
        .footer {position:fixed; left:0; bottom:0; width:100%;
                 text-align:center; font-size:0.9rem; color:#6b7280;
                 padding:8px 0;}
        </style>
        <div class="footer">© 2025 <a href="https://www.linkedin.com/in/lavanya-srivastava"
             target="_blank">Lavanya Srivastava</a> · All Rights Reserved</div>
        """,
        unsafe_allow_html=True
    )

    user_question = st.text_input("Ask a question about your PDFs")
    if user_question:
        if not os.path.isdir(INDEX_DIR):
            st.error("No index found. Please upload PDFs and click 'Submit & Process' first.")
        else:
            try:
                answer_question(user_question)
            except Exception as e:
                st.error(f"Error answering: {e}")

    with st.sidebar:
        st.title("Menu")
        st.caption("App by **Lavanya Srivastava**")
        pdf_docs = st.file_uploader("Upload PDFs, then click 'Submit & Process'",
                                    accept_multiple_files=True, type=["pdf"])
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
                        st.success("✅ Done! You can now ask questions.")
                    except Exception as e:
                        st.error(f"Failed to process PDFs: {e}")

if __name__ == "__main__":
    main()
