# backend/models/agent_graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from backend.models.intent_classifier import IntentClassifier
from backend.models.symptom_extractor import SymptomExtractor
from backend.models.orchestrator import Orchestrator
from backend.models.rag_chain import RAGHealthAssistant
import os
from typing_extensions import TypedDict

class AgentState(TypedDict, total=False):
    user_query: str
    intent: str
    symptoms: list[str]
    final_answer: dict


class AgenticGraph:

    def __init__(self, api_key: str =None, model_path: str =None):
        self.intent_classifier = IntentClassifier()
        self.symptom_extractor = SymptomExtractor()
        self.orchestrator = Orchestrator()
        self.rag = RAGHealthAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # LangGraph definition
        graph = StateGraph(
            AgentState,
            input_schema=AgentState,
            output_schema=AgentState
        )

        graph.add_node("intent", self.intent_node)
        graph.add_node("symptom_parse", self.symptom_node)
        graph.add_node("hybrid_orchestrator", self.hybrid_node)
        graph.add_node("pure_rag", self.pure_rag_node)

        graph.set_entry_point("intent")

        # branching edges
        graph.add_conditional_edges(
            "intent",
            self.intent_router,
            {
                "symptoms": "symptom_parse",
                "general_medical": "pure_rag"
            }
        )

        # graph.add_edge("symptom_parse", "hybrid_orchestrator")
        graph.add_conditional_edges(
            "symptom_parse",
            self.check_symptoms,
            {
                "has_symptoms": "hybrid_orchestrator",
                "no_symptoms": "pure_rag",
            }
        )
        graph.add_edge("hybrid_orchestrator", END)
        graph.add_edge("pure_rag", END)

        self.graph = graph.compile(checkpointer=MemorySaver())

    # ------ Node Implementations ----------

    def intent_node(self, state):
        print("[intent_node] state at entry:", state)
        intent = self.intent_classifier.classify(state["user_query"])
        return {"intent": intent}

    def intent_router(self, state):
        return state["intent"]

    def symptom_node(self, state):
        symptoms = self.symptom_extractor.extract(state["user_query"])
        return {"symptoms": symptoms}

    def check_symptoms(self, state):
        symptoms = state.get("symptoms", [])
        if symptoms and len(symptoms) > 0 :
            return "has_symptoms"
        else:
            return "no_symptoms"

    def hybrid_node(self, state):
        ml_rag_output = self.orchestrator.process_symptom_query(
            state["user_query"],
            state["symptoms"]
        )
        return {"final_answer": ml_rag_output}

    def pure_rag_node(self, state):
        rag_output = self.rag.query(state["user_query"])
        modified_ouput =  {
            "mode": "PURE_RAG_RESPONSE",
            "ml_prediction": None,
            "symptoms": [],
            "rag_query": state["user_query"],
            "rag_answer": rag_output.get("answer"),
            "rag_sources": rag_output.get("sources", []),
            "explanation": "Using pure RAG-based reasoning to answer the user query",
        }
        return {"final_answer": modified_ouput}

    # ------ Public API ----------

    def run(self, user_query: str, session_id : str = None):
        return self.graph.invoke({"user_query": user_query},config={"configurable": {"thread_id": session_id}})
