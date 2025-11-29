# backend/models/agent_graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from backend.models.intent_classifier import IntentClassifier
from backend.models.symptom_extractor import SymptomExtractor
from backend.models.orchestrator import Orchestrator
from backend.models.rag_chain import RAGHealthAssistant
from backend.models.predictor import DiseasePredictor
import os
from pathlib import Path
from typing_extensions import TypedDict
import uuid

class AgentState(TypedDict, total=False):
    user_query: str
    intent: str
    symptoms: list[str]
    history: list
    final_answer: dict


class AgenticGraph:

    def __init__(self, api_key: str =None, model_path: str =None):
        self.memory = {} 
        self.intent_classifier = IntentClassifier()
        self.symptom_extractor = SymptomExtractor()
        self.orchestrator = Orchestrator()
        self.rag = RAGHealthAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))
        MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "predictor.pkl"
        self.predictor = DiseasePredictor(model_path=MODEL_PATH)

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

    def symptom_node(self, state: AgentState):
        # Existing symptoms from session (if any)
        existing_symptoms = state.get("symptoms", [])

        # New symptoms from this query only
        new_symptoms = self.symptom_extractor.extract(state["user_query"])

        merged = []
        seen = set()
        for s in existing_symptoms + new_symptoms:
            if s not in seen:
                seen.add(s)
                merged.append(s)

        return {"symptoms": merged}


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

    def pure_rag_node(self, state: AgentState):
        history = state.get("history")

        # If we have history, use multi-turn RAG
        if history:
            messages = history + [{"role": "user", "content": state["user_query"]}]
            rag_output = self.rag.query_with_history(messages)
        else:
            rag_output = self.rag.query(state["user_query"])

        modified_output = {
            "mode": "PURE_RAG_RESPONSE",
            "ml_prediction": None,
            "symptoms": [],
            "rag_query": state["user_query"],
            "rag_answer": rag_output.get("answer"),
            "rag_sources": rag_output.get("sources", []),
            "explanation": "Using pure RAG-based reasoning to answer the user query",
        }
        return {"final_answer": modified_output}

    # ------ Public API ----------

    def run(self, user_query, session_id=None, history=None):
        """
        Entry point for the agent.
        - Handles session state (symptoms across turns)
        - Delegates all logic to the LangGraph
        """

        if not session_id:
            session_id = str(uuid.uuid4())

        # Initialize memory for this session
        if session_id not in self.memory:
            self.memory[session_id] = {
                "symptoms": [],
                "intent_history": [],
            }

        session_memory = self.memory[session_id]

        # Build initial LangGraph state
        initial_state: AgentState = {
            "user_query": user_query,
            "symptoms": session_memory["symptoms"],  # prior symptoms across turns
        }
        if history:
            initial_state["history"] = history

        # Invoke graph with thread_id-based memory
        config = {"configurable": {"thread_id": session_id}}
        print("[agent_graph] session_id:", session_id)
        result_state = self.graph.invoke(initial_state, config=config)

        # Update session memory with whatever graph produced
        new_symptoms = result_state.get("symptoms", [])
        session_memory["symptoms"] = new_symptoms
        session_memory["intent_history"].append(result_state.get("intent"))

        # final_answer is produced by hybrid_node or pure_rag_node
        final_answer = result_state["final_answer"]

        return {
            "final_answer": final_answer
        }

