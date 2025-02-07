import sys
import os

# Add src/ directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from retrieval import HybridRetriever
from reranking import Reranker
from summarization import GeminiSummarizer

class RAGPipeline:
    def __init__(self):
        """
        Initialize the RAG pipeline with retrieval, reranking, and summarization modules.
        """
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.summarizer = GeminiSummarizer()

    def run(self, query: str, query_embedding: list, top_k: int = 5):
        """
        Execute the full RAG pipeline: retrieve, rerank, and summarize.
        :param query: User query.
        :param query_embedding: Embedding of the query (for vector search).
        :param top_k: Number of top results to retrieve and rerank.
        :return: Final summarized response.
        """
        print("Retrieving top candidates...")
        retrieved_results = self.retriever.retrieve(query, query_embedding, top_k)

        # Prepare candidates for reranking (e.g., get text for the chunks)
        combined_results = [
            {"chunk_file": r["chunk_file"], "text": self.retriever.read_chunk_text(r["chunk_file"])}
            for r in retrieved_results["bm25_results"]
        ]

        print("Reranking candidates...")
        reranked_results = self.reranker.rerank(query, combined_results)

        print("Summarizing results...")
        summary = self.summarizer.summarize(reranked_results)

        return summary


# Example Usage
if __name__ == "__main__":
    pipeline = RAGPipeline()

    # Example query and query embedding
    query = "What documents are required for a health insurance claim?"
    query_embedding = [0.1] * 384  # Replace with actual embedding generation logic

    # Run the pipeline
    summary = pipeline.run(query, query_embedding, top_k=5)
    print("Final Summary:", summary)
