# backend/models/symptom_extractor.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import ast
import os

class SymptomExtractor:

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract only medical symptoms from the text. "
             "Return them as a Python list of snake_case strings. "
             "Example: ['weight_loss', 'nausea', 'muscle_pain']"),
            ("user", "{query}")
        ])

    def extract(self, query: str):
        chain = self.prompt | self.llm
        output = chain.invoke({"query": query}).content.strip()
        return ast.literal_eval(output)
