"""
Rule-based Triage Engine for MediLens AI.

Executes PRE-LLM and returns triage classification in < 5ms.
No external API calls or ML inference.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class TriageResult:
    level: int
    category: str
    reason: str


# --- Triage Rule Tables ---
# Ordered from most critical (1) to least (5).
# Each entry: (level, category, keywords_list, reason_template)

TRIAGE_RULES = [
    # Level 1 — Resuscitation
    (1, "Resuscitation", [
        "unconscious", "unconsciousness", "cardiac arrest", "no pulse",
        "not breathing", "stopped breathing", "severe seizure", "anaphylaxis",
        "anaphylactic shock", "drowning", "unresponsive"
    ], "Life-threatening symptom detected. Immediate medical resuscitation required."),

    # Level 2 — Emergency
    (2, "Emergency", [
        "difficulty breathing", "cannot breathe", "can't breathe",
        "breathing difficulty", "shortness of breath", "severe chest pain",
        "chest pain", "confusion", "severely confused", "severe dehydration",
        "high fever with confusion", "stroke", "sudden numbness", "sudden weakness",
        "coughing blood", "vomiting blood", "severe allergic reaction"
    ], "Serious symptoms that require rapid medical evaluation. Visit emergency care."),

    # Level 3 — Urgent
    (3, "Urgent", [
        "fever above 102", "fever over 102", "high fever", "severe pain",
        "severe body pain", "persistent vomiting", "rash with fever",
        "rash and fever", "infected wound", "deep cut", "head injury",
        "broken bone", "fracture", "moderate dehydration", "seizure history"
    ], "Symptoms require medical attention. Not immediately life-threatening but should be evaluated promptly."),

    # Level 4 — Less Urgent
    (4, "Less Urgent", [
        "moderate fever", "fever 100", "fever 101", "fever 102",
        "mild infection", "localized rash", "ear pain", "sore throat",
        "mild vomiting", "nausea", "mild diarrhea", "urinary discomfort",
        "mild headache", "minor sprain", "minor burn"
    ], "Mild to moderate symptoms that should be evaluated but are not an emergency."),

    # Level 5 — Non-Urgent
    (5, "Non-Urgent", [
        "minor skin irritation", "mild itching", "dry skin", "dandruff",
        "minor rash", "cold symptoms", "runny nose", "mild cough",
        "no fever", "insect bite", "sunburn", "routine checkup"
    ], "Minor symptoms that can usually be managed with basic home care."),
]


def classify_triage(symptoms: str) -> TriageResult:
    """
    Classify the triage level of patient symptoms using keyword rules.
    Returns the highest-urgency match found, or Level 5 as default.
    Runs in < 5 ms (pure string operations, no ML).
    """
    symptoms_lower = symptoms.lower()

    for level, category, keywords, reason in TRIAGE_RULES:
        for keyword in keywords:
            if keyword in symptoms_lower:
                # Return matched reason with matched keyword highlighted
                detailed_reason = f"{reason} (Triggered by: '{keyword}')"
                return TriageResult(
                    level=level,
                    category=category,
                    reason=detailed_reason
                )

    # Default fallback if no keywords matched
    return TriageResult(
        level=5,
        category="Non-Urgent",
        reason="No emergency or urgent keywords detected. Symptoms appear minor and manageable with basic care."
    )


def normalize_confidence(raw_score: float, rank: int = 0) -> float:
    """
    Normalize a FAISS L2 distance score to a confidence value between 0.5 and 0.9.
    
    - Lower L2 distance = higher similarity = higher confidence.
    - rank: position in the result list (0 = best match, degrades by 0.04 per rank).
    - Result is always clamped to [0.5, 0.9].
    """
    # Convert L2 distance to a 0-1 similarity score
    # Empirically, scores in [0, 2] map well; beyond 2 is poor match
    similarity = max(0.0, 1.0 - (raw_score / 2.0))
    
    # Scale to [0.5, 0.9] range
    normalized = 0.5 + (similarity * 0.4)
    
    # Penalize lower-ranked results slightly
    normalized = normalized - (rank * 0.04)
    
    # Clamp to [0.5, 0.9]
    return round(max(0.5, min(0.9, normalized)), 2)
