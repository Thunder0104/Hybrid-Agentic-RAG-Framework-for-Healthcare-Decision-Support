# Hybrid Agentic–RAG Healthcare Decision Support ChatBot

This project implements a Hybrid Agentic–Retrieval Augmented Generation (RAG) healthcare decision support system. The system combines a symptom-based prediction agent with vector-based medical knowledge retrieval to provide context-aware and explainable healthcare insights.

The architecture integrates a trained prediction model with multiple vector databases built from symptom data and encyclopedic medical content, enabling reliable and interpretable responses.

---

## Prerequisites

- Python 3.9 or higher  
- Node.js (v16 or higher recommended)  
- npm  
- OpenAI API key

---

## Backend Setup

### Install Python Dependencies

`pip install -r requirements.txt`

---

## Environment Configuration

Set the OpenAI API key before running training scripts.

Windows (Command Prompt):

`set OPENAI_API_KEY=<your_api_key>`

The API key value is available in apikey.txt.

---

## Preprocessing Steps (Required)

Run the following commands after installing dependencies and setting the API key.

### Train the Prediction Agent(for first time)

`python -m backend.training.train_predictor`

### Build Symptom Vector Database(for first time)

`python -m backend.training.build_vectorstore_symptom`

### Build Encyclopedia / PDF Vector Database(for first time)

`python -m backend.training.build_vectorstore_pdf`

---

## Running the Backend Server

`uvicorn backend.main:app --reload`

Backend runs at `http://127.0.0.1:8000`

---

## Frontend Setup

In a new terminal:

cd frontend  
npm install  
npm run start  

Frontend runs at http://localhost:3000

---

## System Workflow

1. User asks a question in the chatbot interface
2. The prediction agent identifies probable diseases  
3. RAG retrieves relevant medical context  
4. System generates explainable responses

---

## Disclaimer

For academic and research purposes only. Not for clinical use.
