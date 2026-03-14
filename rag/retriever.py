import faiss
import numpy as np
import os
import pickle
import wikipedia
from typing import List, Optional, Dict, Any

class Retriever:
    def __init__(self, index_path="vector_db/faiss_index.bin", meta_path="vector_db/meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.metadata = []
        self.dimension = 384 # Default dimension for all-MiniLM-L6-v2
        self.load_index()

    def build_index(self, embeddings, metadata):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = metadata
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump(metadata, f)

    def load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            print(f"Index not found at {self.index_path}. Please build the index first.")

    def fetch_wikipedia_fallback(self, query: str):
        try:
            # Drop common stopwords for better wiki searches if needed, or just search query
            print(f"Searching Wikipedia for: {query}")
            results = wikipedia.search(query)
            if results:
                print(f"Wikipedia found: {results[0]}")
                summary = wikipedia.summary(results[0], sentences=3)
                return [summary], [f"Wikipedia: {results[0]}"], [1.0]
        except Exception as e:
            print(f"Wikipedia fallback failed: {e}")
            pass
        return [], [], []

    def retrieve(self, query_embedding, query_text: str, top_k=3, fallback_threshold=1.1):
        if not self.index:
            return self.fetch_wikipedia_fallback(query_text)

        # Reshape for faiss search
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        sources = []
        scores = []
        best_score = float('inf')
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                score = float(distances[0][i])
                scores.append(score)
                if score < best_score:
                    best_score = score
                
                meta = self.metadata[idx]
                results.append(meta["text"])
                sources.append(meta["source"])
        
        # Instant RAG Optimization: Disable slow Wikipedia network fallback
        if len(results) == 0:
            print(f"No results found in local database for query: {query_text}")
            
        return results, sources, scores
    def retrieve_agentic(self, embedder, generator, query_text: str, patient_info: Optional[Dict[str, Any]] = None, top_k=10, final_top_k=3):
        """Agentic Retrieval: HyDE -> Vector Search -> LLM Reranking."""
        print(f"[Agentic] Pass 1: Generating HyDE clinical document...")
        hyde_doc = generator.generate_hyde_query(query_text, patient_info)
        
        print(f"[Agentic] Pass 2: Vector Search with HyDE...")
        hyde_embedding = embedder.embed_query(hyde_doc)
        results, sources, scores = self.retrieve(hyde_embedding, query_text, top_k=top_k)
        
        if not results:
            return [], [], []
            
        print(f"[Agentic] Pass 3: Reranking {len(results)} documents...")
        reranked_results = generator.rerank_documents(query_text, results, top_n=final_top_k)
        
        # Map back to original sources
        final_sources = []
        for res in reranked_results:
            for i, orig_res in enumerate(results):
                if res == orig_res:
                    final_sources.append(sources[i])
                    break
        
        return reranked_results, final_sources, scores[:len(reranked_results)]
