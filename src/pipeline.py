import sys
import os

# Add the src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from retrieval import HybridRetriever  # Now it should work
from reranking import Reranker
class RAGPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline with retrieval and reranking modules.
        """
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

    def run(self, query: str, query_embedding: list, top_k: int = 5):
        """
        Perform retrieval and reranking for the RAG pipeline.
        :param query: Query text.
        :param query_embedding: Query embedding for vector search.
        :param top_k: Number of top results to retrieve and rerank.
        :return: Final ranked results.
        """
        print("Retrieving top candidates...")
        retrieved_results = self.retriever.retrieve(query, query_embedding, top_k)

        # Combine BM25 results and Vector Search results (you can customize this logic)
        combined_results = [
            {"chunk_file": r["chunk_file"], "text": self.retriever.read_chunk_text(r["chunk_file"])}
            for r in retrieved_results["bm25_results"]
        ]

        print("Reranking candidates...")
        reranked_results = self.reranker.rerank(query, combined_results)
        return reranked_results


# Example Usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()

    # Example query and query embedding
    query = "What documents are required for a health insurance claim?"
    query_embedding = [0.1] * 384  # Replace with actual embedding

    # Run the pipeline
    results = pipeline.run(query, query_embedding, top_k=5)

    # Display final results
    print("\nFinal Results:")
    for result in results:
        print(f"Chunk: {result['chunk_file']}, Relevance Score: {result['relevance_score']}, Text: {result['text']}")
