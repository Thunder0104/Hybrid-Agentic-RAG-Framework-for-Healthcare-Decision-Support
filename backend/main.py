# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uuid

from backend.config import PREDICTOR_MODEL_PATH

from backend.models.agent_graph import AgenticGraph
from backend.models.intent_classifier import IntentClassifier
from backend.models.symptom_extractor import SymptomExtractor
from backend.models.predictor import DiseasePredictor
from backend.models.orchestrator import Orchestrator
from backend.models.rag_chain import RAGHealthAssistant

import os

app = FastAPI(title="Hybrid Agentic-Rag Backend")

# Initialize all components across backend
agent = AgenticGraph()
intent_classifier = IntentClassifier()
symptom_extractor = SymptomExtractor()
predictor = DiseasePredictor(model_path=PREDICTOR_MODEL_PATH)
orchestrator = Orchestrator()
rag = RAGHealthAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))

# AGENT RUN ENDPOINT

class AskRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None


@app.post("/ask")
def ask_api(req: AskRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = agent.run(req.user_query, session_id=session_id)

    return {
        "session_id": session_id,
        "final_answer": result["final_answer"]
    }

# Component APIs/ENDPOINTS

class TextPayload(BaseModel):
    text: str


@app.post("/debug/intent")
def debug_intent(req: TextPayload):
    intent = intent_classifier.classify(req.text)
    return {"intent": intent}


@app.post("/debug/symptoms")
def debug_symptoms(req: TextPayload):
    symptoms = symptom_extractor.extract(req.text)
    return {"symptoms": symptoms}


class PredictorPayload(BaseModel):
    symptoms: List[str]


@app.post("/debug/predict")
def debug_predict(req: PredictorPayload):
    prediction = predictor.predict(req.symptoms)
    return prediction


class OrchestratorPayload(BaseModel):
    user_query: str
    symptoms: List[str]


@app.post("/debug/orchestrator")
def debug_orchestrator(req: OrchestratorPayload):
    result = orchestrator.process_symptom_query(
        req.user_query,
        req.symptoms
    )
    return result

# PURE RAG ENDPOINT

class RagQueryPayload(BaseModel):
    query: str


@app.post("/rag/query")
def rag_query(req: RagQueryPayload):
    result = rag.query(req.query)
    return result

# SESSION MANAGEMENT ENDPOINTS

@app.post("/session/start")
def session_start():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


class SessionEndPayload(BaseModel):
    session_id: str


@app.post("/session/end")
def session_end(req: SessionEndPayload):
    # MemorySaver auto-cleans entries per thread_id
    return {"status": "terminated", "session_id": req.session_id}
