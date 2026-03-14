import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List

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
                # Use 'models/gemini-flash-latest' which is the most stable alias in the list
                self.model = genai.GenerativeModel("gemini-2.5-flash")
            except Exception as e:
                with open("/tmp/gemini_error.log", "a") as f:
                    f.write(f"Initialization Error: {e}\n")
                print(f"Failed to initialize Gemini: {e}")
                self.test_mode = True

    def generate_hyde_query(self, symptoms: str, patient_info: Optional[Dict[str, Any]] = None) -> str:
        """Agentic Pass 1: Generate a hypothetical document to improve vector search (HyDE)."""
        if self.test_mode:
            return f"Hypothetical medical case for {symptoms}"
        
        patient_block = ""
        if patient_info:
            patient_block = f"Patient: {patient_info.get('age')}yo {patient_info.get('gender')}. "

        prompt = f"""{patient_block}Symptoms: {symptoms}
        
        Write a brief hypothetical clinical note or medical article passage that describes a patient with these symptoms. 
        Focus on technical medical terminology that would likely appear in a medical textbook or clinical study.
        Do not include a diagnosis, just the clinical presentation. Keep it under 150 words."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[HyDE] Generation failed: {e}")
            return symptoms

    def rerank_documents(self, query: str, documents: List[str], top_n: int = 3) -> List[str]:
        """Agentic Pass 2: Use LLM to select the most relevant documents from the initial retrieval."""
        if self.test_mode or not documents:
            return documents[:top_n]
            
        import re, json
        
        doc_list = ""
        for i, doc in enumerate(documents):
            doc_list += f"DOC [{i}]: {doc[:500]}...\n\n"
            
        prompt = f"""Query: {query}
        
        Below are several medical document snippets. Select the {top_n} documents that are MOST relevant and provide the most accurate medical evidence for the query.
        Return ONLY the indices of the selected documents in a JSON list format, e.g., [0, 2, 5].
        
        Documents:
        {doc_list}"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            match = re.search(r'\[\s*\d+.*\]', text)
            if match:
                indices = json.loads(match.group())
                selected = []
                for idx in indices:
                    if isinstance(idx, int) and idx < len(documents):
                        selected.append(documents[idx])
                return selected[:top_n]
        except Exception as e:
            print(f"[Reranker] Failed: {e}")
            
        return documents[:top_n]

    def verify_assessment(self, draft: str, sources: List[str]) -> str:
        """Agentic Pass 4: Audit Agent pass to ensure the analysis is accurate based on sources."""
        if self.test_mode or not sources:
            return draft
            
        source_text = "\n\n".join(sources)
        prompt = f"""Draft Assessment:
        {draft}
        
        Medical Sources:
        {source_text}
        
        Review the draft assessment against the provided medical sources. 
        If there are any contradictions or inaccuracies based ON THE SOURCES, fix them. 
        Ensure the output remains helpful and does not provide a definitive diagnosis.
        Return the final verified assessment only."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"[Verifier] Failed: {e}")
            return draft

    def generate(
        self,
        context: str,
        symptoms: str,
        env_context: str = "",
        patient_info: Optional[Dict[str, Any]] = None,
    ):
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

        prompt = f"""
You are an AI medical triage assistant.

Rules:
- Do NOT give a medical diagnosis.
- Use only the provided medical context.
- If information is uncertain, say "possible".
- Be concise and medically accurate.

PATIENT
Name: {patient_info.get('name','N/A') if patient_info else "N/A"}
Age: {patient_info.get('age','N/A') if patient_info else "N/A"}
Gender: {patient_info.get('gender','N/A') if patient_info else "N/A"}

SYMPTOMS
{symptoms}

ENVIRONMENT
{env_context}

MEDICAL CONTEXT
{context}

TRIAGE SCALE
1 = Resuscitation (Critical)
2 = Emergency
3 = Urgent
4 = Less Urgent
5 = Non-Urgent

OUTPUT JSON ONLY

{{
  "possible_conditions": [
    {{"name": "condition", "confidence": 0.0}}
  ],
  "explanation": "medical explanation",
  "advice": "care steps",
  "triage_level": 1,
  "triage_category": "category"
}}
"""
        
        import json, re, time

        last_error = None
        for attempt in range(3):   # up to 3 attempts with exponential backoff
            try:
                # Ensure we are using a stable model name
                response = self.model.generate_content(prompt)
                
                if not response or not response.text:
                    raise Exception("Empty response from Gemini")
                    
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
                with open("/tmp/gemini_error.log", "a") as f:
                    f.write(f"Generation Error (Attempt {attempt+1}): {e}\n")
                
                if "403" in error_str:
                    print(f"\n[CRITICAL ERROR] Gemini API Key is invalid or leaked (403).")
                    print("Please get a new key from https://aistudio.google.com/ and update .env\n")
                    break # Don't retry on 403
                elif "429" in error_str or "quota" in error_str.lower():
                    print(f"\n[CRITICAL ERROR] Gemini API Key Quota Exceeded (429).")
                    print("Please check your billing details or use a new key from https://aistudio.google.com/\n")
                    break # Don't retry on 429 quota issues to avoid massive delays
                else:
                    print(f"[Generator] Gemini error (attempt {attempt+1}/3): {e}")
                    # If it's a safety block or other terminal error, don't necessarily break yet but log it
                    if "finish_reason: SAFETY" in error_str:
                        print("[Generator] Response blocked by safety filters.")
                        break

        # Final fallback
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

    def generate_report(
        self,
        context: str,
        symptoms: str,
        image_analysis: str = None,
        patient_info: Optional[Dict[str, Any]] = None,
    ):
        if self.test_mode:
            return "# Medical Assessment Report\n\nThis is a mock report generated in Local Test Mode."

        # Build patient demographics block
        patient_block = ""
        if patient_info:
            patient_block = f"""
Patient Demographics:
- Name: {patient_info.get('name', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}
"""

        prompt = f"""You are an AI medical triage assistant helping generate a preliminary clinical assessment report.

Your task is to analyze the patient's symptoms, observations, and retrieved medical context, and produce a structured medical report that includes an emergency triage classification.

IMPORTANT RULES:
- Do NOT provide a definitive medical diagnosis.
- Provide only a preliminary AI-based assessment.
- Use clear, professional, and medically understandable language.
- The output must be structured so it can be converted directly into a PDF medical report.
- Consider the patient's age and gender when assessing conditions and risks.

---

PATIENT INFORMATION
{patient_block}
Symptoms:
{symptoms}

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
Brief summary of the patient's symptoms, age, gender, and observations.

## 2. Observed Symptoms
List symptoms clearly in bullet points including pain intensity and duration.

## 3. Possible Medical Conditions
List 2–4 possible conditions that may match the symptoms.
For each condition include:
- Condition name
- Short explanation
- Confidence level (Low / Moderate / High)
- Any age or gender-specific relevance

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
Explain why the symptoms match the possible conditions, considering the patient's age and gender.

## 6. Recommended Immediate Care
Provide safe precautions or home-care steps appropriate for the patient's age and gender.

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
