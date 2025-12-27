# backend/training/build_vectorstore_pdf.py

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from backend.config import DATA_DIR


PDF_FOLDER = "data/encyclopedias"
VECTOR_DB_PATH = "vectorstore/encyclopedia_store"

def build_pdf_vectorstore():
    print("Building PDF Encyclopedia Vectorstore...")

    # Load PDFs
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    all_docs = []

    for file in pdf_files:
        path = PDF_FOLDER / file
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} pages from {len(pdf_files)} PDFs")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)

    # Embed and save
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    vectordb.persist()

    print(f"Encyclopedia vectorstore created at: {VECTOR_DB_PATH}")

if __name__ == "__main__":
    build_pdf_vectorstore()
