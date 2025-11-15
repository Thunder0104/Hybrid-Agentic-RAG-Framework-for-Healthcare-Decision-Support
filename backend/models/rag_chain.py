# backend/models/rag_chain.py

from pathlib import Path
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class RAGHealthAssistant:
    def __init__(self,
                 symptom_store_path="vectorstore/symptom_store",
                 encyclopedia_store_path="vectorstore/encyclopedia_store",
                 llm_model="gpt-4o-mini",
                 symptom_weight=0.6,
                 encyclopedia_weight=0.4):
        """
        Hybrid RAG Assistant that retrieves from:
        1. Symptom store (CSV knowledge)
        2. Encyclopedia store (PDF knowledge)
        and generates medical explanations with preventive advice.
        """

        print("Initializing RAGHealthAssistant...")

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 1. Load vectorstores
        self.symptom_store = Chroma(
            persist_directory=symptom_store_path,
            embedding_function=self.embeddings
        )

        self.encyclopedia_store = Chroma(
            persist_directory=encyclopedia_store_path,
            embedding_function=self.embeddings
        )

        # 2. Create retrievers
        self.symptom_retriever = self.symptom_store.as_retriever(search_kwargs={"k": 3})
        self.encyclopedia_retriever = self.encyclopedia_store.as_retriever(search_kwargs={"k": 3})

        # 3️. Combine both retrievers (weighted)
        self.retriever = EnsembleRetriever(
            retrievers=[self.symptom_retriever, self.encyclopedia_retriever],
            weights=[symptom_weight, encyclopedia_weight]
        )

        # 4️. Initialize LLM (you can swap this for a local model)
        # self.llm = ChatOpenAI(model=llm_model, temperature=0.3)

        from transformers import pipeline
        from langchain_huggingface import HuggingFacePipeline
        generator = pipeline("text-generation", model="google/flan-t5-large", max_new_tokens=256)
        self.llm = HuggingFacePipeline(pipeline=generator)

        # from transformers import pipeline
        # from langchain_huggingface import HuggingFacePipeline

        # generator = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=256)
        # self.llm = HuggingFacePipeline(pipeline=generator)

        # generator = pipeline(
        #     "text-generation",
        #     model="mistralai/Mistral-7B-Instruct-v0.2",
        #     max_new_tokens=512,
        #     temperature=0.4,
        #     device_map="auto"
        # )
        # self.llm = HuggingFacePipeline(pipeline=generator)


        # 5. Create the RAG prompt
        prompt_template = """
        You are an intelligent healthcare assistant combining structured and unstructured knowledge.
        Use the provided context from medical encyclopedias and symptom databases to answer accurately.

        If the user asks about a disease or symptom:
        - Explain what it is
        - Describe causes and symptoms
        - Provide 3-4 precautions or treatment recommendations if available
        - Maintain a clear, factual, and non-diagnostic tone

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # 6. Define RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

        print("RAGHealthAssistant ready.")

    def query(self, question: str):
        """Query the hybrid retriever (symptom + encyclopedia)."""
        print(f"Querying RAG system for: {question}")
        # response = self.qa_chain(question)
        response = self.qa_chain.invoke(question)
        answer = response["result"]
        sources = [doc.metadata.get("source", "") for doc in response["source_documents"]]
        return {"answer": answer, "sources": sources}


# Quick test
if __name__ == "__main__":
    rag = RAGHealthAssistant()
    query = "What precautions should I take for diabetes?"
    result = rag.query(query)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:\n", result["sources"])
