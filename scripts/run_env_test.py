import requests
import json

BASE = "http://localhost:8000/rag/analyze"

tests = [
    {
        "label": "TEST 1 — No location (backward compat)",
        "payload": {
            "symptoms": "mild cough and runny nose",
            "image_analysis": None
        }
    },
    {
        "label": "TEST 2 — With New Delhi coordinates",
        "payload": {
            "symptoms": "itchy red rash on arm with mild fever",
            "image_analysis": "red inflamed rash",
            "latitude": 28.6139,
            "longitude": 77.2090
        }
    },
]

for t in tests:
    print(f"\n{'='*60}\n{t['label']}")
    r = requests.post(BASE, json=t["payload"])
    d = r.json()
    print(f"  triage_level              : {d.get('triage_level')}")
    print(f"  triage_category           : {d.get('triage_category')}")
    print(f"  triage_reason             : {d.get('triage_reason')}")
    print(f"  environmental_risk_assessment: {d.get('environmental_risk_assessment')}")
    conds = d.get("possible_conditions", [])
    for c in conds:
        print(f"  condition: {c.get('name')} | confidence: {c.get('confidence')}")
