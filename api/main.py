import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Ensure the parent directory is in the system path so "rag" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict, Union

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
        # Fast Embedding Model: (all-MiniLM-L6-v2) loaded once globally
        embedder = Embedder()
        # FAISS Index Preloading: Index loaded once globally
        retriever = Retriever(
            index_path=os.path.join(base_dir, "vector_db", "faiss_index.bin"),
            meta_path=os.path.join(base_dir, "vector_db", "meta.pkl")
        )
        generator = GeminiGenerator()
        print("RAG components initialized successfully. (Preloaded Index)")
    except Exception as e:
        print(f"Error initializing RAG components: {e}")


# ─── Request / Response Models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    """New backend payload — matches the structure sent by the MediLens backend."""
    name: str
    age: int
    gender: str
    symptoms: Optional[Union[List[str], str]] = None
    symptoms_name: Optional[str] = None
    pain_intensity: int = Field(..., ge=1, le=10, description="Pain severity 1 (low) – 10 (high)")
    symptom_duration: str = Field(..., description="How long symptom has persisted, e.g. '3 days'")
    additional_notes: Optional[str] = None


class QueryResponse(BaseModel):
    possible_conditions: List[Dict[str, Any]]
    explanation: str
    advice: str
    sources: List[str]
    triage_level: int = 0
    triage_category: str = ""
    triage_reason: str = ""
    environmental_risk_assessment: str = ""
    note: Optional[str] = None


class ReportResponse(BaseModel):
    report: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_bare_symptoms(req: QueryRequest) -> str:
    """Combine ONLY the symptom fields into one query string for normalization."""
    parts = []
    
    # Handle 'symptoms' field (List or Str)
    if req.symptoms:
        if isinstance(req.symptoms, list):
            parts.append(", ".join(req.symptoms))
        else:
            parts.append(req.symptoms)
            
    # Handle 'symptoms_name' field
    if req.symptoms_name:
        parts.append(req.symptoms_name)
        
    # If both are missing, use a fallback
    if not parts:
        parts.append("unknown symptoms")

    return ". ".join(parts)

def _build_full_query(req: QueryRequest, bare_symptoms: str) -> str:
    """Append context fields like pain and duration to the normalized symptoms."""
    parts = [bare_symptoms]
    if req.pain_intensity:
        parts.append(f"pain intensity {req.pain_intensity} out of 10")
    if req.symptom_duration:
        parts.append(f"duration {req.symptom_duration}")
    if req.additional_notes:
        parts.append(req.additional_notes)
    return ". ".join(parts)


def _patient_info(req: QueryRequest) -> Dict[str, Any]:
    """Extract patient demographic dict for context building."""
    return {
        "name": req.name,
        "age": req.age,
        "gender": req.gender,
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "RAG service running"}


def sync_retrieval(symptoms: str):
    enhancer = QueryEnhancer()
    enhanced_query = enhancer.enhance(symptoms)
    query_embedding = embedder.embed_query(enhanced_query)
    retrieved_docs, sources, scores = retriever.retrieve(query_embedding, enhanced_query)
    return retrieved_docs, sources, scores, enhanced_query


@app.post("/rag/analyze", response_model=QueryResponse)
async def analyze_symptoms(request: QueryRequest):
    if not embedder or not retriever or not generator:
        raise HTTPException(status_code=500, detail="RAG system not initialized properly.")

    # Build ONLY symptom text for fast normalization
    bare_symptoms = _build_bare_symptoms(request)
    patient = _patient_info(request)

    # Feature 1: Multilingual Symptom Normalization (< 5 ms)
    norm_result = normalize_symptoms(bare_symptoms)
    original_input = norm_result["original_input"]
    normalized_bare = norm_result["normalized_query"]
    
    # Re-attach pain/duration context for the final LLM prompt and search
    full_query = _build_full_query(request, normalized_bare)

    # --- Rule-Based Triage Engine (runs BEFORE LLM, < 5 ms) ---
    triage = classify_triage(full_query)

    # Feature 5: Safety Escalation — if Level 1 or 2, bypass RAG entirely
    if triage.level <= 2:
        return QueryResponse(
            possible_conditions=[{"name": "Medical Emergency", "confidence": 0.9}],
            explanation=f"Emergency triage classification: {triage.category}. {triage.reason}",
            advice="Please seek immediate medical attention or call emergency services (911 / local equivalent) RIGHT NOW.",
            sources=["Triage Safety Engine"],
            triage_level=triage.level,
            triage_category=triage.category,
            triage_reason=triage.reason,
        )

    # Retrieval Caching
    cache_key = full_query.strip().lower()
    if cache_key in query_cache:
        print("Returning cached result.")
        return query_cache[cache_key]

    try:
        # Instant RAG Optimization: Bypass Agentic Passes for fast sync_retrieval
        retrieval_task = asyncio.to_thread(sync_retrieval, full_query)
        (retrieved_docs, sources, scores, _) = await retrieval_task

        # Build context with high-relevance chunks
        context = build_context(retrieved_docs, full_query, patient_info=patient)

        # Generate response using Gemini (patient demographics injected into prompt)
        llm_response = await asyncio.to_thread(
            generator.generate, context, original_input, patient_info=patient
        )

        explanation = ""
        advice = ""
        possible_conditions = []
        raw_conditions = []
        note = None

        if isinstance(llm_response, dict):
            explanation = llm_response.get("explanation", "")
            advice = llm_response.get("advice", "")
            raw_conditions = llm_response.get("possible_conditions", [])
            
            # Check if it's the fallback template
            if not raw_conditions and "could not generate a detailed analysis" in explanation.lower():
                note = "AI Analysis is currently in Fallback Mode. Check server logs for API key or quota issues."
        else:
            explanation = str(llm_response)

        # Feature 3: Confidence Score Normalization (0.5 – 0.9 from FAISS scores)
        best_dist = min(scores) if scores else 1.0
        for i, cond in enumerate(raw_conditions):
            conf_val = normalize_confidence(best_dist, rank=i)
            if isinstance(cond, str):
                possible_conditions.append({"name": cond, "confidence": conf_val})
            elif isinstance(cond, dict):
                cond["confidence"] = conf_val
                possible_conditions.append(cond)

        # Feature 4: Triage + RAG Integration
        response = QueryResponse(
            possible_conditions=possible_conditions,
            explanation=explanation,
            advice=advice,
            sources=list(set(sources)),
            triage_level=triage.level,
            triage_category=triage.category,
            triage_reason=triage.reason,
            environmental_risk_assessment="No location data provided.",
            note=note
        )

        query_cache[cache_key] = response
        return response
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/report", response_model=ReportResponse)
async def generate_medical_report(request: QueryRequest):
    if not embedder or not retriever or not generator:
        raise HTTPException(status_code=500, detail="RAG system not initialized properly.")

    composite_symptoms = _build_composite_query(request)
    patient = _patient_info(request)
    symptoms_lower = composite_symptoms.lower()

    # Safety Filter
    dangerous_keywords = ["severe pain", "bleeding", "unconsciousness", "difficulty breathing", "chest pain"]
    if any(danger in symptoms_lower for danger in dangerous_keywords):
        return ReportResponse(
            report="# Medical Emergency Detected\n\nDangerous symptoms detected. Please seek immediate medical attention or call emergency services."
        )

    try:
        retrieval_task = asyncio.to_thread(sync_retrieval, composite_symptoms)
        (retrieved_docs, sources, scores, enhanced_query) = await retrieval_task

        context = build_context(retrieved_docs, enhanced_query, patient_info=patient)

        sources_str = "\\n".join([f"- {s}" for s in set(sources)])
        context_with_sources = f"Knowledge Base Sources:\\n{sources_str}\\n\\n{context}"

        llm_response = await asyncio.to_thread(
            generator.generate_report,
            context_with_sources,
            composite_symptoms,
            patient_info=patient,
        )

        return ReportResponse(report=llm_response)
    except Exception as e:
        print(f"Error processing report request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
