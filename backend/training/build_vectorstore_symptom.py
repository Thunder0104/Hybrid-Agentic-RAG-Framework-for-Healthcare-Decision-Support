# backend/training/build_vectorstore_symptom.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.config import DATA_DIR

# Paths
SYMPTOM_DESC_PATH = DATA_DIR / "symptom_description.csv"
SYMPTOM_PRECAUTION_PATH = DATA_DIR / "symptom_precaution.csv"
VECTOR_DB_PATH = "vectorstore/symptom_store"

def build_symptom_vectorstore():
    print("Building Symptom Vectorstore...")

    # Load datasets
    desc_df = pd.read_csv(SYMPTOM_DESC_PATH)
    prec_df = pd.read_csv(SYMPTOM_PRECAUTION_PATH)

    # Merge on Disease
    merged_df = pd.merge(desc_df, prec_df, on="Disease", how="outer")

    # Combine description + precautions
    merged_df.fillna("", inplace=True)

    docs = []
    for _, row in merged_df.iterrows():
        disease = row["Disease"]
        description = row.get("Description", "")
        precautions = ", ".join(
            str(row.get(f"Precaution_{i}", "")) for i in range(1, 5) if str(row.get(f"Precaution_{i}", "")) != ""
        )

        text = f"Disease: {disease}\nDescription: {description}\nPrecautions: {precautions}"
        docs.append(Document(page_content=text, metadata={"source": "symptom_store", "disease": disease}))

    print(f"Loaded {len(docs)} combined disease records")

    # Split if needed (long text safety)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    # splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in Chroma
    vectordb = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=VECTOR_DB_PATH)
    vectordb.persist()

    print(f"Symptom vectorstore created at: {VECTOR_DB_PATH}")

if __name__ == "__main__":
    build_symptom_vectorstore()
