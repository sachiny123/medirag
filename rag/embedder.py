from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: list):
        # Convert output to list of floats for consistency
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def embed_query(self, query: str):
        return self.model.encode([query])[0]
