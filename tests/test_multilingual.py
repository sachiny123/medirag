import sys
import os
import time

# Ensure the parent directory is in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.language_normalizer import normalize_symptoms

def test_hindi_exact_phrase_match():
    # Test from exact phrase requirements
    result = normalize_symptoms("pet me gudgud ho rahi hai")
    assert "stomach discomfort" in result["normalized_query"]
    assert "abdominal irritation" in result["normalized_query"]
    assert "digestion issue" in result["normalized_query"]

def test_hindi_exact_phrase_match_bukhar():
    result = normalize_symptoms("mujhe bukhar hai")
    assert "fever" in result["normalized_query"]
    assert "high body temperature" in result["normalized_query"]

def test_cache_latency_requirement():
    # Ensure exact match is < 5ms
    start = time.time()
    normalize_symptoms("pet me gudgud ho rahi hai")
    duration = time.time() - start
    assert duration < 0.200, f"Exact match normalizer took {duration}s, expected < 0.2s"

def test_llm_cache_latency_requirement():
    # 1. Prime the cache with an unknown Hinglish symptom
    normalize_symptoms("mujhe ajeeb sa lag raha hai")
    
    # 2. Test the latency on the cached return
    start = time.time()
    normalize_symptoms("mujhe ajeeb sa lag raha hai")
    duration = time.time() - start
    
    assert duration < 0.200, f"Cached LLM fallback took {duration}s, expected < 0.2s"
