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

# Define output schema for the API
class QueryResponse(BaseModel):
    query: str
    summary: str  # Final summarized output from the pipeline

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline and return the final summarized response.
    """
    query = request.query
    top_k = request.top_k

    # Generate a placeholder query embedding (replace with actual embedding logic if needed)
    query_embedding = [0.1] * 384  # Replace with an actual query embedding

    # Run the pipeline
    print(f"Processing query: {query}")
    summary = pipeline.run(query, query_embedding, top_k=top_k)

    # Return the final summarized response
    return {
        "query": query,
        "summary": summary,
    }
