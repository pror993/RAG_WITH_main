from sentence_transformers import SentenceTransformer
import os
import json
from typing import List, Dict


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Hugging Face embedding model.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def process_chunks(self, chunk_dir: str) -> List[Dict[str, List[float]]]:
        """
        Generate embeddings for all text chunks in a directory.
        Saves the chunk filename and its embedding.
        """
        embeddings = []
        for chunk_file in os.listdir(chunk_dir):
            if chunk_file.endswith(".txt"):
                with open(os.path.join(chunk_dir, chunk_file), "r") as f:
                    text = f.read()
                    embedding = self.generate_embeddings([text])[0]  # Single embedding per chunk
                    embeddings.append({"chunk_file": chunk_file, "embedding": embedding})
        return embeddings

    def save_embeddings(self, embeddings: List[Dict[str, List[float]]], output_file: str):
        """
        Save embeddings as a JSON file.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(embeddings, f)
        print(f"Saved {len(embeddings)} embeddings to {output_file}")


# Example Usage
if __name__ == "__main__":
    # Directories for input and output
    chunk_dir = "./processed_chunks/"  # Directory containing text chunks
    output_file = "./embeddings/embeddings.json"  # Output file for embeddings

    # Initialize and run the embedding process
    generator = EmbeddingGenerator()
    embeddings = generator.process_chunks(chunk_dir)
    generator.save_embeddings(embeddings, output_file)
