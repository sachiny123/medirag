def build_context(retrieved_docs: list, symptoms: str, image_analysis: str = None):
    docs_text = "\n---\n".join(retrieved_docs)
    
    context = (
        f"Retrieved Medical Passages:\n{docs_text}\n\n"
        f"User Symptoms: {symptoms}"
    )
    
    if image_analysis:
        context += f"\nImage Analysis: {image_analysis}"
        
    return context
