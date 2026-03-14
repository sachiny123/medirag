import requests
import json

BASE = "http://localhost:8000/rag/analyze"

tests = [
    {
        "label": "TEST 1 — Level 4: Localized rash + moderate fever",
        "payload": {"symptoms": "localized rash on arm with moderate fever 101", "image_analysis": "red inflamed rash"}
    },
    {
        "label": "TEST 2 — Level 3: Rash + high fever + severe body pain",
        "payload": {"symptoms": "rash with fever and severe body pain", "image_analysis": None}
    },
    {
        "label": "TEST 3 — Level 1: Emergency escalation (unconscious)",
        "payload": {"symptoms": "patient is unconscious and not breathing", "image_analysis": None}
    },
    {
        "label": "TEST 4 — Level 2: Chest pain escalation",
        "payload": {"symptoms": "severe chest pain with difficulty breathing", "image_analysis": None}
    },
]

for t in tests:
    print("\n" + "=" * 60)
    print(t["label"])
    r = requests.post(BASE, json=t["payload"])
    d = r.json()
    print(f"  triage_level    : {d.get('triage_level')}")
    print(f"  triage_category : {d.get('triage_category')}")
    print(f"  triage_reason   : {d.get('triage_reason')}")
    conds = d.get("possible_conditions", [])
    for c in conds:
        print(f"  condition       : {c.get('name')} | confidence: {c.get('confidence')}")
