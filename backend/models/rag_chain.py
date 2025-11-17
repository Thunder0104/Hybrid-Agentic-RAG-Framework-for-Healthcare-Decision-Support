# backend/models/rag_chain.py

from langchain_openai import OpenAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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


# Quick standalone test
if __name__ == "__main__":
    import os

    rag = RAGHealthAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))
    result = rag.query("What precautions should I take for diabetes?")
    print("\nAnswer:\n", result["answer"])
    print("\nSources:\n", result["sources"])
