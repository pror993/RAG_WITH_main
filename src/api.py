import sys
import os

# Add src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.pipeline import RAGPipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
# Initialize FastAPI app
app = FastAPI()

# Initialize RAG pipeline
pipeline = RAGPipeline()

# Define input schema for the API
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Optional; default to top 5 results

# Define output schema for the API (optional)
class QueryResponse(BaseModel):
    chunk_file: str
    relevance_score: float
    summary: str

@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline and return ranked results.
    """
    query = request.query
    top_k = request.top_k

    # Generate a placeholder query embedding (replace with actual embedding logic if needed)
    query_embedding = [0.1] * 384  # Replace with actual embedding generated from query

    # Run the pipeline
    print(f"Processing query: {query}")
    results = pipeline.run(query, query_embedding, top_k=top_k)

    # Prepare response
    response = []
    for result in results:
        response.append({
            "chunk_file": result["chunk_file"],
            "relevance_score": result["relevance_score"],
            "summary": result.get("summary", result["text"])  # Use summarized text if available
        })

    return response
