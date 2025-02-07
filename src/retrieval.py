from pymilvus import connections, Collection
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from dotenv import load_dotenv
import json
import numpy as np


# Load environment variables
load_dotenv()

class HybridRetriever:
    def __init__(self, collection_name="document_embeddings"):
        """
        Initialize Hybrid Retrieval with BM25 and Milvus vector search.
        """
        self.collection_name = collection_name
        self.vectorizer = TfidfVectorizer()  # BM25 approximation

        # Connect to Milvus
        uri = os.getenv("MILVUS_PUBLIC_ENDPOINT")
        api_key = os.getenv("MILVUS_API_KEY")
        print(f"Connecting to Milvus at {uri}...")
        connections.connect(
            alias="default",
            uri=uri,
            token=api_key
        )

        # Check connection and load collection
        self.collection = self._load_collection()

        # Load chunk metadata for BM25
        with open("./embeddings/embeddings.json", "r") as f:
            self.chunk_metadata = json.load(f)
        self.chunks = [metadata["chunk_file"] for metadata in self.chunk_metadata]

        # Fit BM25 vectorizer on chunk text
        print("Fitting BM25 vectorizer...")
        chunk_texts = [self.read_chunk_text(chunk_file) for chunk_file in self.chunks]
        self.vectorizer.fit(chunk_texts)

    def _load_collection(self):
        """
        Load a collection from Milvus. Raises an exception if the collection doesn't exist.
        """
        try:
            collection = Collection(self.collection_name)
            collection.load()
            print(f"Collection '{self.collection_name}' loaded successfully.")
            return collection
        except Exception as e:
            print(f"Error loading collection '{self.collection_name}': {e}")
            raise e

    def read_chunk_text(self, chunk_file):
        """
        Read the text of a chunk from the processed_chunks folder.
        """
        with open(f"./processed_chunks/{chunk_file}", "r") as f:
            return f.read()
    def bm25_search(self, query, top_k=5):
        """
        Perform BM25 keyword-based retrieval.
        """
        query_vector = self.vectorizer.transform([query])
        chunk_vectors = self.vectorizer.transform([self.read_chunk_text(c) for c in self.chunks])

        # Compute scores as dot product
        scores = (chunk_vectors @ query_vector.T).toarray().flatten()
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [{"chunk_file": self.chunks[i], "bm25_score": scores[i]} for i in ranked_indices]

    def vector_search(self, query_embedding, top_k=5):
        """
        Perform vector similarity search using Milvus.
        """
        search_params = {"nprobe": 10}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
        )
        return [{"chunk_id": result.id, "distance": result.distance} for result in results[0]]

    def retrieve(self, query, query_embedding, top_k=5):
        """
        Perform hybrid retrieval: BM25 + Vector Search.
        """
        print("Performing BM25 search...")
        bm25_results = self.bm25_search(query, top_k)

        print("Performing vector similarity search...")
        vector_results = self.vector_search(query_embedding, top_k)

        # Combine results (you can customize this combination logic)
        combined_results = {
            "bm25_results": bm25_results,
            "vector_results": vector_results,
        }
        return combined_results


# Example Usage
if __name__ == "__main__":
    # Initialize HybridRetriever
    retriever = HybridRetriever()

    # Example query
    query = "What documents are required for a health insurance claim?"
    query_embedding = [0.1] * 384  # Replace with the actual query embedding

    # Perform hybrid retrieval
    results = retriever.retrieve(query, query_embedding, top_k=5)

    # Display results
    print("\nBM25 Results:")
    for r in results["bm25_results"]:
        print(f"Chunk: {r['chunk_file']}, BM25 Score: {r['bm25_score']}")

    print("\nVector Results:")
    for r in results["vector_results"]:
        print(f"Chunk ID: {r['chunk_id']}, Distance: {r['distance']}")
