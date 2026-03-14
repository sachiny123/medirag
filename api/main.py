import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Ensure the parent directory is in the system path so "rag" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Any, Dict

from rag.embedder import Embedder
from rag.retriever import Retriever
from rag.context_builder import build_context
from rag.generator import GeminiGenerator
from rag.enhancer import QueryEnhancer
from rag.triage_engine import classify_triage, normalize_confidence
from rag.environment_service import get_environment_data, build_env_context_block, summarize_env_risk
from rag.language_normalizer import normalize_symptoms

app = FastAPI(title="MediLens AI RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embedder = None
retriever = None
generator = None
query_cache = {}

@app.on_event("startup")
def startup_event():
    global embedder, retriever, generator
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # 7. Fast Embedding Model: (all-MiniLM-L6-v2) loaded once globally
        embedder = Embedder()
        # 6. FAISS Index Preloading: Index loaded once globally
        retriever = Retriever(
            index_path=os.path.join(base_dir, "vector_db", "faiss_index.bin"),
            meta_path=os.path.join(base_dir, "vector_db", "meta.pkl")
        )
        generator = GeminiGenerator()
        print("RAG components initialized successfully. (Preloaded Index)")
    except Exception as e:
        print(f"Error initializing RAG components: {e}")

class QueryRequest(BaseModel):
    symptoms: str
    image_analysis: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# Response Model — PDF-compatible structure
class QueryResponse(BaseModel):
    possible_conditions: List[Dict[str, Any]]
    explanation: str
    advice: str
    sources: List[str]
    triage_level: int = 0
    triage_category: str = ""
    triage_reason: str = ""
    environmental_risk_assessment: str = ""

class ReportResponse(BaseModel):
    report: str

@app.get("/health")
def health_check():
    return {"status": "RAG service running"}

def sync_retrieval(symptoms: str):
    # 1. Query Enhancement
    enhancer = QueryEnhancer()
    enhanced_query = enhancer.enhance(symptoms)
    query_embedding = embedder.embed_query(enhanced_query)
    retrieved_docs, sources, scores = retriever.retrieve(query_embedding, enhanced_query)
    return retrieved_docs, sources, scores, enhanced_query

def sync_image_process(image_analysis: str):
    # Mock CPU bound image processing
    return image_analysis

@app.post("/rag/analyze", response_model=QueryResponse)
async def analyze_symptoms(request: QueryRequest):
    if not embedder or not retriever or not generator:
        raise HTTPException(status_code=500, detail="RAG system not initialized properly.")
        
    symptoms_lower = request.symptoms.lower()

    # Feature 1: Multilingual Symptom Normalization (< 200 ms)
    norm_result = normalize_symptoms(request.symptoms)
    original_input = norm_result["original_input"]
    normalized_query = norm_result["normalized_query"]

    # --- Rule-Based Triage Engine (runs BEFORE LLM, < 5ms) ---
    # Using normalized_query so Hindi phrases map to English keywords
    triage = classify_triage(normalized_query)

    # Feature 5: Safety Escalation — if Level 1 or 2, bypass RAG entirely
    if triage.level <= 2:
        return QueryResponse(
            possible_conditions=[{"name": "Medical Emergency", "confidence": 0.9}],
            explanation=f"Emergency triage classification: {triage.category}. {triage.reason}",
            advice="Please seek immediate medical attention or call emergency services (911 / local equivalent) RIGHT NOW.",
            sources=["Triage Safety Engine"],
            triage_level=triage.level,
            triage_category=triage.category,
            triage_reason=triage.reason
        )

    # 5. Retrieval Caching
    cache_key = request.symptoms.strip().lower()
    if cache_key in query_cache:
        print("Returning cached result.")
        return query_cache[cache_key]

    try:
        # 9. Parallel Processing — retrieval + image analysis + env data concurrently
        # Feature 2: Query Expansion for RAG - Retrieval uses the expanded matched terminology
        retrieval_task = asyncio.to_thread(sync_retrieval, normalized_query)
        image_task = asyncio.to_thread(sync_image_process, request.image_analysis)

        # Fetch environmental data if coordinates provided
        env_task = None
        if request.latitude is not None and request.longitude is not None:
            env_task = asyncio.to_thread(get_environment_data, request.latitude, request.longitude)

        # Gather tasks — with or without env
        if env_task:
            (retrieved_docs, sources, scores, enhanced_query), processed_image, env_data = await asyncio.gather(
                retrieval_task, image_task, env_task
            )
        else:
            (retrieved_docs, sources, scores, enhanced_query), processed_image = await asyncio.gather(
                retrieval_task, image_task
            )
            env_data = None

        # Build context + optional environmental prompt block
        context = build_context(retrieved_docs, enhanced_query, processed_image)
        env_context_block = build_env_context_block(env_data) if env_data else ""

        # Generate response using Gemini (env context injected into prompt, original input for language style)
        llm_response = await asyncio.to_thread(generator.generate, context, original_input, env_context_block)
        
        # Parse output
        explanation = ""
        advice = ""
        possible_conditions = []
        raw_conditions = []
        triage_level = 0
        triage_category = ""
        
        if isinstance(llm_response, dict):
            explanation = llm_response.get("explanation", "")
            advice = llm_response.get("advice", "")
            raw_conditions = llm_response.get("possible_conditions", [])
            # LLM triage is fallback only — rule-based engine takes precedence
        else:
            explanation = str(llm_response)

        # --- Feature 3: Confidence Score Normalization (0.5 – 0.9 from FAISS scores) ---
        best_dist = min(scores) if scores else 1.0
        for i, cond in enumerate(raw_conditions):
            conf_val = normalize_confidence(best_dist, rank=i)
            if isinstance(cond, str):
                possible_conditions.append({"name": cond, "confidence": conf_val})
            elif isinstance(cond, dict):
                # Always override LLM confidence with normalized FAISS-based value
                cond["confidence"] = conf_val
                possible_conditions.append(cond)

        # --- Feature 4: Triage + RAG Integration ---
        # Rule-based triage ALWAYS wins over LLM triage
        response = QueryResponse(
            possible_conditions=possible_conditions,
            explanation=explanation,
            advice=advice,
            sources=list(set(sources)),
            triage_level=triage.level,
            triage_category=triage.category,
            triage_reason=triage.reason,
            environmental_risk_assessment=summarize_env_risk(env_data) if env_data else "No location data provided."
        )
        
        # Save to cache
        query_cache[cache_key] = response
        return response
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/report", response_model=ReportResponse)
async def generate_medical_report(request: QueryRequest):
    if not embedder or not retriever or not generator:
        raise HTTPException(status_code=500, detail="RAG system not initialized properly.")
        
    symptoms_lower = request.symptoms.lower()
    
    # 8. Safety Filter
    dangerous_keywords = ["severe pain", "bleeding", "unconsciousness", "difficulty breathing", "chest pain"]
    if any(danger in symptoms_lower for danger in dangerous_keywords):
        return ReportResponse(
            report="# Medical Emergency Detected\n\nDangerous symptoms detected. Please seek immediate medical attention or call emergency services."
        )

    try:
        # Run retrieval and image analysis concurrently
        retrieval_task = asyncio.to_thread(sync_retrieval, request.symptoms)
        image_task = asyncio.to_thread(sync_image_process, request.image_analysis)
        
        (retrieved_docs, sources, scores, enhanced_query), processed_image = await asyncio.gather(retrieval_task, image_task)

        # Build context
        context = build_context(retrieved_docs, enhanced_query, processed_image)
        
        # Inject explicit source lists back into context for Gemini to extract
        sources_str = "\\n".join([f"- {s}" for s in set(sources)])
        context_with_sources = f"Knowledge Base Sources:\\n{sources_str}\\n\\n{context}"

        # Generate report using Gemini
        llm_response = await asyncio.to_thread(
            generator.generate_report, 
            context_with_sources, 
            request.symptoms, 
            request.image_analysis
        )
        
        return ReportResponse(report=llm_response)
    except Exception as e:
        print(f"Error processing report request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
