import sys
import os
import json
import argparse
import requests

# Add parent dir to path so we can import rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.context_builder import build_context
from rag.generator import GeminiGenerator

def test_local_pipeline(query="itchy red rash on arm with mild fever"):
    print("\n--- Testing RAG Pipeline Modules Locally ---")
    
    patient_info = {"name": "Test User", "age": 30, "gender": "male"}
    
    print(f"Querying: '{query}'")
    print(f"Patient: {patient_info}")
    
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        embedder = Embedder()
        query_embedding = embedder.embed_query(query)
        
        retriever = Retriever(
            index_path=os.path.join(base_dir, "vector_db", "faiss_index.bin"),
            meta_path=os.path.join(base_dir, "vector_db", "meta.pkl")
        )
        docs, sources, scores = retriever.retrieve(query_embedding, query)
        
        context = build_context(docs, query, patient_info=patient_info)
        print("\nContext built successfully.")
        
        generator = GeminiGenerator()
        response = generator.generate(context, query, patient_info=patient_info)
        
        print("\nPipeline Result:")
        print(json.dumps(response, indent=2))
        print("Sources:", sources)
        print("------------------------------------------\n")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")

def test_api(query="headache and mild fever"):
    print("\n--- Testing API Endpoint ---")
    url = "http://127.0.0.1:8001/rag/analyze"
    payload = {
        "name": "Test User",
        "age": 30,
        "gender": "male",
        "symptoms_name": query,
        "pain_intensity": 5,
        "symptom_duration": "2 days",
        "additional_notes": "Started after getting wet in rain"
    }
    
    print(f"Sending API Query: {json.dumps(payload, indent=2)}")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("API Response:")
        print(json.dumps(response.json(), indent=2))
        
        print("\n--- Testing API Report Endpoint ---")
        report_url = "http://127.0.0.1:8001/rag/report"
        report_response = requests.post(report_url, json=payload)
        report_response.raise_for_status()
        print("Report API Response:")
        print(report_response.json().get("report", ""))

    except requests.exceptions.ConnectionError:
        print("Failed to connect to API. Make sure the server is running on 172.20.10.10:8001")
    except Exception as e:
        print(f"API request failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the RAG pipeline.")
    parser.add_argument("--api", action="store_true", help="Test the API endpoint instead of the local pipeline.")
    parser.add_argument("--query", type=str, default="itchy red rash on arm with mild fever", help="The medical query to search for.")
    args = parser.parse_args()
    
    if args.api:
        test_api(args.query)
    else:
        test_local_pipeline(args.query)
