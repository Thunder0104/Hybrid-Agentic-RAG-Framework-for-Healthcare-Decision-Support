# backend/models/intent_classifier.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

class IntentClassifier:
    """
    Uses GPT to classify user intent:
    - "symptoms"
    - "general_medical"
    """

    def __init__(self):

        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "Classify the user query into exactly one category:\n"
            "1. symptoms -> if the user describes bodily problems, discomfort, or health complaints.\n"
            "2. general_medical -> if the user asks for explanations, definitions, or facts.\n"
            "Return ONLY the category name."
            ),
            ("user", "{query}")
        ])

    def classify(self, query: str) -> str:
        chain = self.prompt | self.llm
        return chain.invoke({"query": query}).content.strip().lower()
