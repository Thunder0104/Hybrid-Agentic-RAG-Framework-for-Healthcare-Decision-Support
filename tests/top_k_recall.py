import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from collections import defaultdict

# ============================================================
# 1. GOLDEN DATASET FORMAT
# ============================================================

"""
Golden dataset format:

[
    {
        "query": "What precautions should I take for diabetes?",
        "expected_sources": [
             "symptom_store (disease: Diabetes)",
             "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 326)"
        ]
    },
    ...
]
"""

golden_dataset = [
    {
        "query": "What precautions should I take for diabetes?",
        "expected_sources": [
            "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 326)",
            "symptom_store (disease: Diabetes)",
        ]
    },
    {
        "query": "What are symptoms of hypertension?",
        "expected_sources": [
            "symptom_store (disease: Hypertension)"
        ]
    },
]

# ============================================================
# 2. SOURCE FORMATTING
# ============================================================

def format_sources(source_documents: List[Dict[str, Any]]) -> List[str]:
    sources = []
    for doc in source_documents:
        meta = doc["metadata"]
        if "disease" in meta:
            sources.append(f"{meta.get('source')} (disease: {meta.get('disease')})")
        elif "page" in meta:
            sources.append(f"{meta.get('source')} (page: {meta.get('page')})")
        else:
            sources.append(meta.get("source"))
    return sources

# ============================================================
# 3. TOP-K RECALL METRIC
# ============================================================
"""
Definition:

Recall@K = (# retrieved sources that match expected sources) / (total expected sources)
"""

def top_k_recall(expected: List[str], retrieved: List[str], k: int) -> float:
    retrieved_top_k = retrieved[:k]
    print(f"retrieved_top_k->{retrieved_top_k}")
    print(f"expected->{expected}")
    matches = sum(1 for src in expected if src in retrieved_top_k)
    return matches / len(expected)

# ============================================================
# 4. EVALUATION LOOP
# ============================================================

def evaluate_retriever(golden_dataset, retriever_function, k_values=[1, 3, 5]):
    results = []

    for item in golden_dataset:
        query = item["query"]
        expected_sources = item["expected_sources"]

        # Call your retriever function
        response = retriever_function(query)

        # Format sources the same way your RAG pipeline does
        formatted_sources = response["sources"]

        print(formatted_sources)

        # Compute Recall@K for each K
        row = {"query": query}
        for k in k_values:
            row[f"Recall@{k}"] = top_k_recall(expected_sources, formatted_sources, k)
        results.append(row)

    return pd.DataFrame(results)

# ============================================================
# 5. Real Retriever using FastAPI backend
# ============================================================

import requests

API_URL = "http://localhost:8000/api/rag/query"

def real_retriever(query: str):
    payload = {"query": query}
    res = requests.post(API_URL, json=payload).json()

    answer = res["answer"]
    sources = res["sources"]

    return {
        "answer": answer,
        "sources": sources
    }

# ============================================================
# 6. RUN EVALUATION
# ============================================================

df_results = evaluate_retriever(golden_dataset, real_retriever)

print("\n=== Evaluation Results Table ===")
print(df_results)

# ============================================================
# 7. PLOT RECALL CURVE
# ============================================================

def plot_recall_curve(df_results):
    ks = [int(col.split("@")[1]) for col in df_results.columns if col.startswith("Recall@")]
    recall_means = [df_results[f"Recall@{k}"].mean() for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, recall_means, marker="o")
    plt.title("Retriever Performance: Recall@K Curve")
    plt.xlabel("K")
    plt.ylabel("Average Recall")
    plt.grid(True)
    plt.show()


plot_recall_curve(df_results)
