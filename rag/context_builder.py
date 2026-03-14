from typing import Optional, Dict, Any


def build_context(
    retrieved_docs: list,
    symptoms: str,
    image_analysis: str = None,
    patient_info: Optional[Dict[str, Any]] = None,
):
    """Build the context block that is injected into the LLM prompt.

    Parameters
    ----------
    retrieved_docs : list
        Passages retrieved from FAISS / Wikipedia.
    symptoms : str
        Enhanced symptom query string.
    image_analysis : str, optional
        Image analysis text (kept for backward compatibility).
    patient_info : dict, optional
        Patient demographics: ``{"name": ..., "age": ..., "gender": ...}``
    """
    docs_text = "\n---\n".join(retrieved_docs)

    context = (
        f"Retrieved Medical Passages:\n{docs_text}\n\n"
        f"User Symptoms: {symptoms}"
    )

    if patient_info:
        context += (
            f"\n\nPatient Profile:"
            f"\n- Name: {patient_info.get('name', 'N/A')}"
            f"\n- Age: {patient_info.get('age', 'N/A')}"
            f"\n- Gender: {patient_info.get('gender', 'N/A')}"
        )

    if image_analysis:
        context += f"\nImage Analysis: {image_analysis}"

    return context
