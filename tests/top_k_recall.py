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
    "query": "What are key symptoms and causes of Acoustic neuroma?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 58)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 59)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 60)"
    ]
  },
  {
    "query": "What is Adjustment disorders and how is it diagnosed?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 94)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 94)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 95)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Alzheimer\u2019s disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 3371)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1157)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 170)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Aortic valve stenosis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 398)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1008)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 399)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Atherosclerosis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 468)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 469)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 470)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Barbiturate-induced coma?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 521)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 520)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 522)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Black lung disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 591)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 591)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 592)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Brucellosis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 702)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 702)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 704)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Cat-scratch disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 249)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 774)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 776)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Chagas\u2019 disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 835)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 836)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 837)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Chronic granulomatous disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 3920)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 916)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 917)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Coronary artery disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1046)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1047)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1048)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Diabetic ketoacidosis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1185)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 4214)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1193)"
    ]
  },
  {
    "query": "What is Ehlers-Danlos syndrome and how is it diagnosed?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1284)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1288)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1285)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Epididymitis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1381)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1381)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1383)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Failure to thrive?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1454)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1453)",
    ]
  },
  {
    "query": "What are key symptoms and causes of Flesh-eating disease?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1516)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1516)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1516)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Gastroenteritis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1595)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1595)",
    ]
  },
  {
    "query": "What are key symptoms and causes of Glomerulonephritis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1653)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2632)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1653)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Hairy cell leukemia?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1698)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1698)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1700)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Hemochromatosis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1775)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1772)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1772)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Hepatitis C?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2272)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1805)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1805)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Hives?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1846)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1847)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1847)"
    ]
  },
  {
    "query": "What is Hypercoagulation disorders and how is it diagnosed?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1910)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1910)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1915)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Hypocalcemia?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1954)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1942)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1942)"
    ]
  },
  {
    "query": "What is Idiopathic primary renal hematuric/proteinuric syndrome and how is it diagnosed?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1987)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1987)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 1989)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Ischemia?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2101)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2101)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2102)"
    ]
  },
  {
    "query": "What is Kidney cancer and what are common symptoms?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2146)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2146)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2148)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Laryngitis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2198)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2198)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 3973)"
    ]
  },
  {
    "query": "What are key symptoms and causes of Listeriosis?",
    "expected_sources": [
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2261)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2260)",
      "encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf (page: 2262)"
    ]
  }
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
