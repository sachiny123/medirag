# MediLens AI RAG Microservice

This is the hybrid RAG microservice for MediLens AI. It uses FastAPI for the API layer,
FAISS and sentence-transformers for local retrieval, Google Gemini for generation,
and Wikipedia as a fallback via hybrid retrieval.

## Setup and Installation

1. Navigate to the `rag_service` directory:
   ```bash
   cd d:\MEDILENS\rag_service
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   Copy `.env.example` to a new file named `.env` and configure your API key.
   ```bash
   cp .env.example .env
   # Windows
   # copy .env.example .env
   ```

## Running the Application

### 1. Build the FAISS Vector Index
Before running the API, build the vector database using the sample datasets.
```bash
python scripts/build_index.py
```

### 2. Local Testing (Optional)
Test the RAG pipeline locally (bypassing the server) using test_mode fallback if no API key is provided:
```bash
python scripts/test_rag.py
```

### 3. Start the FastAPI Server
```bash
uvicorn api.main:app
```
Alternatively:
```bash
python -m uvicorn api.main:app
```

The server will be available at `http://localhost:8000`.
You can test the RAG service using the `/rag/analyze` POST endpoint or by running:
```bash
python scripts/test_rag.py --api
```
