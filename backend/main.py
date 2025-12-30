# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
_components = {}
app = FastAPI(title="Hybrid Agentic-Rag Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all components across backend
def get_components():
    if not _components:
        _components["agent"] = AgenticGraph()
        _components["intent_classifier"] = IntentClassifier()
        _components["symptom_extractor"] = SymptomExtractor()
        _components["predictor"] = DiseasePredictor(model_path=PREDICTOR_MODEL_PATH)
        _components["orchestrator"] = Orchestrator()
        _components["rag"] = RAGHealthAssistant(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    return _components

# AGENT RUN ENDPOINT

class AskRequest(BaseModel):
    user_query: str
    history: list = []
    session_id: Optional[str] = None

componenets = get_components()
@app.post("/api/ask")
def ask_api(req: AskRequest):
    session_id = req.session_id or str(uuid.uuid4())
    #components = get_components()
    agent = components["agent"]
    result = agent.run(
        user_query=req.user_query,
        session_id=session_id,
        history=req.history,
    )

    return {
        "session_id": session_id,
        "answer": result["final_answer"],
    }


# Component APIs/ENDPOINTS

class TextPayload(BaseModel):
    text: str


@app.post("/api/debug/intent")
def debug_intent(req: TextPayload):
    #components = get_components()
    intent_classifier = components["intent_classifier"]
    intent = intent_classifier.classify(req.text)
    return {"intent": intent}


@app.post("/api/debug/symptoms")
def debug_symptoms(req: TextPayload):
    #components = get_components()
    symptom_extractor = components["symptom_extractor"]
    symptoms = symptom_extractor.extract(req.text)
    return {"symptoms": symptoms}


class PredictorPayload(BaseModel):
    symptoms: List[str]


@app.post("/api/debug/predict")
def debug_predict(req: PredictorPayload):
    #components = get_components()
    predictor = components["predictor"]
    prediction = predictor.predict(req.symptoms)
    return prediction


class OrchestratorPayload(BaseModel):
    user_query: str
    symptoms: List[str]


@app.post("/api/debug/orchestrator")
def debug_orchestrator(req: OrchestratorPayload):
    #components = get_components()
    orchestrator = components["orchestrator"]

    result = orchestrator.process_symptom_query(
        req.user_query,
        req.symptoms
    )
    return result

# PURE RAG ENDPOINT

class RagQueryPayload(BaseModel):
    query: str


@app.post("/api/rag/query")
def rag_query(req: RagQueryPayload):
    #components = get_components()
    rag = components["rag"]
    result = rag.query(req.query)
    return result

# SESSION MANAGEMENT ENDPOINTS

@app.post("/api/session/start")
def session_start():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


class SessionEndPayload(BaseModel):
    session_id: str


@app.post("/api/session/end")
def session_end(req: SessionEndPayload):
    # MemorySaver auto-cleans entries per thread_id
    return {"status": "terminated", "session_id": req.session_id}
