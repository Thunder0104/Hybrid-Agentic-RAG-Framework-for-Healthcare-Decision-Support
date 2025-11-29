# backend/models/rag_chain.py

from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


class RAGHealthAssistant:
    def __init__(
        self,
        symptom_store_path="vectorstore/symptom_store",
        encyclopedia_store_path="vectorstore/encyclopedia_store",
        openai_api_key=None,
        symptom_weight=0.3,
        encyclopedia_weight=0.7,
    ):
        """
        Hybrid RAG assistant uses:
        - Symptom store (CSV knowledge)
        - Encyclopedia store (PDF knowledge)
        - OpenAI GPT models for answer generation
        """

        print("Initializing RAGHealthAssistant...")

        self.api_key = openai_api_key

        # Embeddings (for Chroma)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load symptom vectorstore
        self.symptom_store = Chroma(
            persist_directory=symptom_store_path,
            embedding_function=self.embeddings,
        )

        # Load encyclopedia vectorstore
        self.encyclopedia_store = Chroma(
            persist_directory=encyclopedia_store_path,
            embedding_function=self.embeddings,
        )

        # Two retrievers with k=3
        self.symptom_retriever = self.symptom_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        self.encyclopedia_retriever = self.encyclopedia_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        # Combine both retrievers
        self.retriever = EnsembleRetriever(
            # retrievers=[self.symptom_retriever, self.encyclopedia_retriever],
            # weights=[symptom_weight, encyclopedia_weight],
            retrievers=[self.encyclopedia_retriever, self.symptom_retriever],
            weights=[encyclopedia_weight, symptom_weight],
        )

        # Prompt template
        prompt_template = """
You are an intelligent medical assistant. Use the context to answer accurately.
If the user asks about a disease or symptom:
- Explain what it is
- Provide symptoms, causes, risks
- Provide 3-4 precautions or treatment recommendations
- Be factual, safe, and helpful

Context:
{context}

Question:
{question}

Answer:
        """

        self.prompt = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        # LLM (OpenAI)
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=self.api_key,
            max_tokens= 1024
        )

        # Build RetrievalQA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )

        print("RAGHealthAssistant ready.\n")

    def query(self, question: str):
        """Run a RAG query through OpenAI with both retrievers."""
        print(f"Querying RAG system for: {question}\n")

        response = self.qa_chain.invoke({"query": question})
        # print(response)
        answer = response["result"]
        # sources = [doc.metadata.get("source", "") for doc in response["source_documents"]]
        sources = []
        for doc in response["source_documents"]:
            meta = doc.metadata
            if "disease" in meta:
                sources.append(f"{meta.get('source')} (disease: {meta.get('disease')})")
            elif "page" in meta:
                sources.append(f"{meta.get('source')} (page: {meta.get('page')})")
            else:
                sources.append(meta.get("source"))

        return {"answer": answer, "sources": sources}

    def query_with_history(self, messages: list):
        """
        Multi-turn RAG retrieval using the existing RetrievalQA chain.
        Leverages:
        - Ensemble retriever (Chroma stores)
        - LangChain prompt
        - OpenAI llm
        """

        # ------------------------------------
        # 1. Build a full conversation context
        # ------------------------------------
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        # ------------------------------------
        # 2. Extract last user message
        # ------------------------------------
        last_user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user_msg = m["content"]
                break

        # Safety check
        if not last_user_msg:
            return {"answer": "", "sources": []}

        # ------------------------------------
        # 3. Inject history into the question before sending to RAG
        # ------------------------------------
        augmented_query = f"""
    Previous conversation:
    {full_context}

    User's latest question:
    {last_user_msg}
        """

        # ------------------------------------
        # 4. Use the existing QC chain
        # ------------------------------------
        response = self.qa_chain.invoke({"query": augmented_query})

        # Extract answer & sources
        answer = response["result"]

        sources = []
        for doc in response["source_documents"]:
            meta = doc.metadata
            if "disease" in meta:
                sources.append(f"{meta.get('source')} (disease: {meta.get('disease')})")
            elif "page" in meta:
                sources.append(f"{meta.get('source')} (page: {meta.get('page')})")
            else:
                sources.append(meta.get("source"))

        return {
            "answer": answer,
            "sources": sources,
        }


    def _retrieve_documents(self, query: str, k: int = 5):
        """
        Retrieve top-k documents using FAISS similarity search.
        """
        if not query.strip():
            return []

        # 1. Embed the user query
        query_embedding = self.embeddings.embed_query(query)

        # 2. Search FAISS index
        scores, indices = self.faiss_index.search(
            np.array([query_embedding]).astype("float32"),
            k
        )

        docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.documents[idx]  # assuming list of dicts
            docs.append({
                "content": doc["content"],
                "source": doc.get("source", f"doc_{idx}"),
                "score": float(score)
            })

        return docs

    def _rerank_documents(self, docs, chat_context):
        """
        Re-rank documents using semantic similarity to the entire conversation context.
        """

        if not docs:
            return docs

        # Embed full chat context
        context_emb = self.embeddings.embed_query(chat_context)

        reranked = []
        for d in docs:
            doc_emb = self.embeddings.embed_query(d["content"])
            similarity = self._cosine_similarity(context_emb, doc_emb)
            reranked.append({**d, "rerank_score": similarity})

        # Sort by rerank score desc
        reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)

        return reranked

    def _cosine_similarity(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def _generate_answer(self, context: str, docs: list):
        """
        Generate an LLM answer using conversation context + documents.
        """

        combined_sources = "\n\n".join(
            [f"[{i+1}] {d['content']}" for i, d in enumerate(docs)]
        )

        prompt = f"""
    You are a clinical decision-support assistant.
    Use only the provided medical evidence.
    Explain findings carefully and safely.

    ### Conversation Context:
    {context}

    ### Retrieved Medical Knowledge:
    {combined_sources}

    ### Task:
    Provide a medically accurate explanation using the above sources.
    """

        response = self.llm.invoke(prompt)
        return response

# Quick standalone test
if __name__ == "__main__":
    import os

    rag = RAGHealthAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))
    result = rag.query("What precautions should I take for diabetes?")
    print("\nAnswer:\n", result["answer"])
    print("\nSources:\n", result["sources"])
