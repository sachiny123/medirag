import time
import functools
import re
from rag.generator import GeminiGenerator

# Rapid Exact Match Dictionary for common Hindi/Hinglish terms (<5ms latency)
COMMON_SYMPTOM_MAP = {
    "pet me gudgud ho rahi hai": "stomach discomfort abdominal irritation digestion issue",
    "bukhar hai": "fever",
    "mujhe bukhar hai": "fever high body temperature",
    "sar dard ho raha hai": "headache",
    "khansi ho rahi hai": "cough",
    "pet dard": "stomach ache abdominal pain",
    "ulti aa rahi hai": "nausea vomiting",
    "chakkar aa raha hai": "dizziness vertigo",
    "khujli ho rahi hai": "itching rash skin irritation",
    "gale me dard": "sore throat throat pain",
    "saans lene me dikkat": "difficulty breathing shortness of breath"
}

_generator = None

@functools.lru_cache(maxsize=1000)
def _translate_with_llm(symptom_text: str) -> str:
    """Fallback LLM translation for unknown Hindi/Hinglish phrases. Cached for speed."""
    global _generator
    if _generator is None:
        _generator = GeminiGenerator()
    
    prompt = f"""You are a medical translator.
The user has entered medical symptoms in Hindi, Hinglish, or English.
Translate this into a standardized English medical query suitable for a search engine.
Do not provide advice or conversation, just return the translated English terms.

Input: {symptom_text}
Output:"""

    try:
        if _generator.test_mode:
            return "unknown symptom"
            
        # Call gemini directly since we just need a simple translation string
        response = _generator.model.generate_content(prompt)
        text = response.text.strip()
        # Remove any Markdown or surrounding quotes
        text = re.sub(r'^["\']|["\']$', '', text)
        return text.lower()
    except Exception as e:
        print(f"[LanguageNormalizer] Translation failed: {e}")
        return symptom_text.lower()


def normalize_symptoms(user_input: str) -> dict:
    """
    Detects and converts user symptoms (Hindi/Hinglish/English) into standardized 
    English medical queries before running RAG retrieval.
    Must return dict with original and normalized text.
    Latency goal: < 200ms.
    """
    start_time = time.time()
    
    clean_input = user_input.strip().lower()
    
    # 1. Fast path: Dictionary lookup (< 5ms)
    normalized = COMMON_SYMPTOM_MAP.get(clean_input)
    
    # 2. Slow path: LLM translation (cached via lru_cache for repeated queries)
    if not normalized:
        normalized = _translate_with_llm(clean_input)
        
    duration = (time.time() - start_time) * 1000
    if duration > 200 and clean_input not in COMMON_SYMPTOM_MAP:
        print(f"[LanguageNormalizer] WARNING: Normalization took {duration:.2f}ms for '{clean_input}'")

    return {
        "original_input": user_input,
        "normalized_query": normalized
    }
