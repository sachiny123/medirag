def chunk_text(text: str, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    
    if text_length <= chunk_size:
        return [text]
        
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_documents(documents, chunk_size=400, overlap=50):
    chunked_docs = []
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "source": f"{doc['source']} (chunk {i})",
            })
    return chunked_docs
