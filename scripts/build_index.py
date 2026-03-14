import sys
import os

# Add parent dir to path so we can import rag modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.loader import get_all_documents
from rag.chunker import process_documents
from rag.embedder import Embedder
from rag.retriever import Retriever
import scripts.ingest_knowledge as ingest_knowledge

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "data")
    
    print("Starting Knowledge Ingestion Pipeline...")
    ingest_knowledge.main()
    
    print("\nLoading documents from all dataset sources...")
    docs = get_all_documents(data_dir)
    print(f"Loaded {len(docs)} documents.")
    
    print("Chunking text...")
    chunked_docs = process_documents(docs)
    print(f"Created {len(chunked_docs)} chunks.")
    
    print("Generating embeddings...")
    embedder = Embedder()
    texts = [doc["text"] for doc in chunked_docs]
    embeddings = embedder.generate_embeddings(texts)
    
    print("Building FAISS index...")
    retriever = Retriever(
        index_path=os.path.join(base_dir, "vector_db", "faiss_index.bin"),
        meta_path=os.path.join(base_dir, "vector_db", "meta.pkl")
    )
    retriever.build_index(embeddings, chunked_docs)
    
    print("Index built successfully!")

if __name__ == "__main__":
    main()
