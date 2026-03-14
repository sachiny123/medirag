import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.test_mode = False
        
        if not self.api_key or self.api_key == "your_api_key_here":
            print("WARNING: Gemini API Key not found. Running in Local Test Mode.")
            self.test_mode = True
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                print(f"Failed to initialize Gemini: {e}")
                self.test_mode = True

    def generate(self, context: str, symptoms: str, env_context: str = ""):
        if self.test_mode:
            explanation = (
                "This is a mock explanation generated in Local Test Mode because the Gemini API key is missing. "
                "Based on the provided symptoms and context, this could be a minor skin condition or viral infection."
            )
            advice = "Please consult a doctor if the symptoms persist. Avoid irritants and rest well."
            
            return {
                "explanation": explanation,
                "advice": advice,
                "possible_conditions": [
                    {"name": "Condition 1 (Mock)", "confidence": 0.85},
                    {"name": "Condition 2 (Mock)", "confidence": 0.70}
                ],
                "triage_level": 4,
                "triage_category": "Less Urgent"
            }

        prompt = f"""You are a medical assistant AI.

Use the provided context to explain possible conditions based on symptoms.
Do not provide a final diagnosis.
Suggest possible causes and basic precautions.
Encourage consulting a doctor if symptoms persist.

MULTILINGUAL SUPPORT:
If the user input symptoms were provided in Hindi or Hinglish, the "explanation" and "advice" fields MUST be returned in simple Hindi. All other structured fields (conditions, categories) MUST remain in English.

Context:
{context}
{env_context}

User Symptoms (Original Input):
{symptoms}

Triage Scale (use this to determine urgency):
- Level 1 — Resuscitation: Life-threatening, requires immediate intervention.
- Level 2 — Emergency: Very serious, requires rapid evaluation.
- Level 3 — Urgent: Needs medical attention, not immediately life-threatening.
- Level 4 — Less Urgent: Mild to moderate, should be evaluated.
- Level 5 — Non-Urgent: Minor, manageable with basic care.

Response Requirements:
Provide a structured JSON response with the following keys:
- "possible_conditions": A list of objects, each containing "name" (string) and "confidence" (float between 0.0 and 1.0).
- "explanation": A detailed explanation of possible causes. If environmental data is present, consider how heat, humidity, or air quality may be worsening symptoms.
- "advice": Basic precautions and advice.
- "triage_level": An integer from 1 to 5 based on the triage scale above.
- "triage_category": The category name string matching the triage level (e.g. "Urgent").
"""
        
        import json, re, time

        last_error = None
        for attempt in range(3):   # up to 3 attempts with exponential backoff
            try:
                response = self.model.generate_content(prompt)
                text = response.text

                # Strip markdown code fences if present (```json ... ```)
                text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
                text = re.sub(r'\s*```$', '', text.strip())

                # Extract JSON object from anywhere in the text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())

                return json.loads(text)

            except Exception as e:
                last_error = e
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    # Extract retry delay from error message if available
                    delay_match = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)
                    wait = int(delay_match.group(1)) if delay_match else (2 ** attempt * 5)
                    wait = min(wait, 30)   # cap at 30 seconds
                    print(f"[Generator] Rate limited. Waiting {wait}s before retry {attempt+1}/3...")
                    time.sleep(wait)
                else:
                    print(f"[Generator] Gemini error (attempt {attempt+1}/3): {e}")
                    break   # non-rate-limit errors won't improve with retry

        # Final fallback — return structured response using raw context
        print(f"[Generator] All retries failed. Using fallback. Last error: {last_error}")
        fallback_explanation = (
            "The AI could not generate a detailed analysis at this time. "
            "Based on the retrieved medical knowledge, the symptoms may be associated with "
            "the conditions listed in the medical context. Please consult a healthcare professional."
        )
        return {
            "explanation": fallback_explanation,
            "advice": "Please consult a doctor for an accurate diagnosis. Rest, stay hydrated, and monitor symptoms closely.",
            "possible_conditions": [],
            "triage_level": 0,
            "triage_category": ""
        }

    def generate_report(self, context: str, symptoms: str, image_analysis: str = None):
        if self.test_mode:
            return "# Medical Assessment Report\n\nThis is a mock report generated in Local Test Mode."

        prompt = f"""You are an AI medical triage assistant helping generate a preliminary clinical assessment report.

Your task is to analyze the patient's symptoms, image observations, and retrieved medical context, and produce a structured medical report that includes an emergency triage classification.

IMPORTANT RULES:
- Do NOT provide a definitive medical diagnosis.
- Provide only a preliminary AI-based assessment.
- Use clear, professional, and medically understandable language.
- The output must be structured so it can be converted directly into a PDF medical report.

---

PATIENT INFORMATION

Symptoms:
{symptoms}

Image Analysis:
{image_analysis if image_analysis else 'None provided'}

Retrieved Medical Context:
{context}

---

TRIAGE SYSTEM

Use the following emergency triage scale to determine urgency:

Level 1 — Resuscitation (Critical Emergency): Life-threatening conditions requiring immediate medical intervention.
Level 2 — Emergency (Very Urgent): Serious symptoms requiring rapid medical evaluation.
Level 3 — Urgent: Symptoms that require medical attention but are not immediately life-threatening.
Level 4 — Less Urgent: Mild to moderate symptoms that should be evaluated but are not severe.
Level 5 — Non-Urgent: Minor symptoms that can usually be managed with basic care.

---

GENERATE A STRUCTURED REPORT IN MARKDOWN FORMAT WITH THE FOLLOWING SECTIONS:

## 1. Case Summary
Brief summary of the patient's symptoms and observations.

## 2. Observed Symptoms
List symptoms clearly in bullet points.

## 3. Possible Medical Conditions
List 2–4 possible conditions that may match the symptoms.
For each condition include:
- Condition name
- Short explanation
- Confidence level (Low / Moderate / High)

## 4. Emergency Triage Assessment
Provide:
- Triage Level (1–5)
- Triage Category Name
- Short explanation of why this level was assigned.

Example format:
**Triage Level:** 3
**Category:** Urgent
**Reason:** Symptoms include rash and fever which may indicate infection and should be evaluated by a healthcare professional.

## 5. Medical Explanation
Explain why the symptoms match the possible conditions.

## 6. Recommended Immediate Care
Provide safe precautions or home-care steps if appropriate.

## 7. When to Seek Medical Attention
List warning signs that require immediate medical care.

## 8. Medical Information Sources
List sources used by the AI system.

## 9. Disclaimer
State clearly that this is an AI-generated preliminary assessment and not a medical diagnosis.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini report generation error: {e}")
            return "Failed to generate report due to an error."

