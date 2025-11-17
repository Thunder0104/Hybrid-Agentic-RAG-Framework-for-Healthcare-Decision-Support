# backend/models/orchestrator.py

from typing import Dict, Any, List
from backend.models.predictor import DiseasePredictor
from backend.models.rag_chain import RAGHealthAssistant
import os
from pathlib import Path

class Orchestrator:
    """
    This orchestrator ONLY handles symptom-based queries.
    The Agentic Router (LangGraph) ensures:
        - Symptom queries -> this orchestrator
        - General medical queries -> pure RAG (skips this class)

    Responsibilities:
        - Accept structured symptoms: ["nausea", "weight_loss", ...]
        - Call ML predictor
        - Use confidence thresholds to determine RAG prompt type
        - Call RAGHealthAssistant
        - Return a unified output
    """

    HIGH_CONF = 0.75
    MID_CONF = 0.40
    
    
    # def __init__(self, api_key: str, model_path: Path = MODEL_PATH):
    def __init__(self):
        # self.api_key = api_key
        MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "predictor.pkl"
        self.predictor = DiseasePredictor(model_path=MODEL_PATH)
        self.rag = RAGHealthAssistant(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # ------------------------------
    # Main Entry for Symptom Queries
    # ------------------------------
    def process_symptom_query(self, user_query: str, symptoms: List[str]) -> Dict[str, Any]:
        """
        Full hybrid pipeline:
            1. Predictor(symptoms)
            2. Confidence-based RAG prompt
            3. Hybrid RAG call
        """

        print(f"[Orchestrator] Symptoms received -> {symptoms}")

        if not symptoms or len(symptoms) == 0:
            # Safety fallback - treat as general query
            rag_query = (
                f"The user describes feeling unwell: \"{user_query}\".\n"
                f"No specific symptoms were identified.\n"
                f"Provide general safe medical guidance, possible causes, "
                f"self-care advice, and when to see a doctor.\n"
            )
            rag_response = self.rag.query(rag_query)
            return {
                "mode": "no_symptoms_detected",
                "ml_prediction": None,
                "symptoms": [],
                "rag_query": rag_query,
                "rag_answer": rag_response.get('answer'),
                "rag_sources": rag_response.get('sources', []),
                "explanation": "Symptoms intent detected, but no extractable symptoms were found. Skipped ML and used pure RAG."
            }

        # ==== 1. Run ML Predictor ====
        ml_result = self.predictor.predict(symptoms)
        conf = ml_result["confidence"]
        print(f"[Orchestrator] ML Prediction: {ml_result}")

        # ==== 2. Select Confidence Mode ====
        if conf > self.HIGH_CONF:
            mode = "high_confidence"
            rag_query = self._build_high_confidence_query(user_query, symptoms, ml_result)

        elif conf >= self.MID_CONF:
            mode = "medium_confidence"
            rag_query = self._build_medium_confidence_query(user_query, symptoms, ml_result)

        else:
            mode = "low_confidence"
            rag_query = self._build_low_confidence_query(user_query, symptoms, ml_result)

        print(f"[Orchestrator] Mode selected: {mode}")
        print(f"[Orchestrator] RAG Query: {rag_query}")

        # ==== 3. Execute RAG ====
        rag_response = self.rag.query(rag_query)

        # ==== 4. Return standard structure ====
        return {
            "mode": mode,
            "ml_prediction": ml_result,
            "symptoms": symptoms,
            "rag_query": rag_query,
            "rag_answer": rag_response.get("answer"),
            "rag_sources": rag_response.get("sources", []),
            "explanation": self._gen_explanation(mode),
        }

    # -----------------------------------------------------------------
    # EXPLANATION TEXT (returned to front-end to describe what happened)
    # -----------------------------------------------------------------
    def _gen_explanation(self, mode: str) -> str:
        if mode == "high_confidence":
            return (
                "ML model is highly confident. Using predicted disease as the primary "
                "interpretation and RAG for medical explanation & precautions."
            )
        if mode == "medium_confidence":
            return (
                "ML model is moderately confident. Using RAG for differential diagnosis "
                "to compare likely possibilities."
            )
        return (
            "ML model is uncertain. Using pure RAG-based reasoning to provide safe, "
            "general medical guidance."
        )

    # ---------------------------
    # CONFIDENCE -> RAG PROMPTS
    # ---------------------------

    def _build_high_confidence_query(self, user_query, symptoms, ml):
        disease = ml["predicted_disease"]
        prob = ml["confidence"]

        return (
            f"The user reports symptoms: {symptoms}\n"
            f"Raw description: \"{user_query}\"\n\n"
            f"An ML model strongly suggests: {disease} (confidence {prob:.2f}).\n"
            f"Assume {disease} is most likely.\n\n"
            f"Please:\n"
            f"1. Explain what {disease} is.\n"
            f"2. Explain how the listed symptoms relate to {disease}.\n"
            f"3. Provide precautions, home care, lifestyle advice.\n"
            f"4. Provide warning signs requiring urgent care.\n"
            f"Do NOT diagnose — provide educational information only."
        )

    def _build_medium_confidence_query(self, user_query, symptoms, ml):
        top_preds = [
            f"{item['disease']} (~{item['probability']:.2f})"
            for item in ml.get("top_predictions", [])
        ]
        top_preds_str = ", ".join(top_preds)

        return (
            f"The user reports symptoms: {symptoms}\n"
            f"Raw description: \"{user_query}\"\n\n"
            f"The ML model is moderately confident.\n"
            f"Possible diseases: {top_preds_str}\n\n"
            f"Perform a differential diagnosis:\n"
            f"- Compare these conditions.\n"
            f"- Explain what symptoms fit which condition.\n"
            f"- Do NOT pick a final diagnosis.\n"
            f"- Provide general guidance and precautions.\n"
        )

    def _build_low_confidence_query(self, user_query, symptoms, ml):
        return (
            f"The user reports symptoms: {symptoms}\n"
            f"Raw description: \"{user_query}\"\n\n"
            f"The ML model has low confidence and may be unreliable.\n"
            f"Ignore the ML predictions.\n\n"
            f"Using trusted medical knowledge:\n"
            f"- Suggest possible categories of issues.\n"
            f"- Provide safe general medical advice.\n"
            f"- Offer precautions.\n"
            f"- Mention red flag symptoms requiring urgent care.\n"
        )
