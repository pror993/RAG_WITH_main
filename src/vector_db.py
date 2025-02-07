from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


class MilvusDB:
    def __init__(self):
        """
        Initialize a connection to Milvus Cloud.
        """
        self.uri = os.getenv("MILVUS_PUBLIC_ENDPOINT")
        self.api_key = os.getenv("MILVUS_API_KEY")
        self.collection_name = "document_embeddings"
        self.dimension = 384  # Embedding dimension (default for MiniLM)

        # Connect to Milvus Cloud
        print(f"Connecting to Milvus at {self.uri}...")
        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.api_key
        )

    def create_collection(self):
        """
        Create a collection in Milvus for storing embeddings.
        """
        # Define schema
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
        ]
        schema = CollectionSchema(fields, description="Storage for document chunk embeddings")

        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection '{self.collection_name}' created with dimension {self.dimension}.")

    def create_index(self):
        """
        Create an index on the 'embedding' field in the collection.
        """
        collection = Collection(self.collection_name)
        index_params = {
            "index_type": "IVF_FLAT",  # Index type
            "metric_type": "L2",      # Similarity metric: L2 (Euclidean Distance) or IP (Inner Product)
            "params": {"nlist": 128}  # Number of clusters (adjust based on dataset size)
        }
        print(f"Creating index on collection '{self.collection_name}'...")
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Index created: {index_params}")

    def insert_embeddings(self, embeddings_file: str):
        """
        Insert embeddings into the Milvus collection.
        :param embeddings_file: Path to the JSON file containing embeddings.
        """
        with open(embeddings_file, "r") as f:
            data = json.load(f)

        # Extract chunk IDs and embeddings
        chunk_ids = [i for i in range(len(data))]
        vectors = [entry["embedding"] for entry in data]

        # Insert into Milvus
        collection = Collection(self.collection_name)
        collection.load()  # Load collection into memory
        collection.insert([chunk_ids, vectors])
        print(f"Inserted {len(vectors)} embeddings into the collection '{self.collection_name}'.")

    def query_embeddings(self, query_vector, top_k: int = 5):
        """
        Perform a similarity search on the embeddings using vector similarity.
        :param query_vector: The vector to search for.
        :param top_k: Number of top results to return.
        """
        collection = Collection(self.collection_name)
        collection.load()

        search_params = {"nprobe": 10}  # Adjust for fine-tuning retrieval performance
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )

        return results


# Example Usage
if __name__ == "__main__":
    # Initialize MilvusDB
    milvus = MilvusDB()

    # Step 1: Create a collection
    milvus.create_collection()

    # Step 2: Create an index
    milvus.create_index()

    # Step 3: Insert embeddings into the collection
    embeddings_file = "./embeddings/embeddings.json"
    milvus.insert_embeddings(embeddings_file)

    # Step 4: Query the collection
    query_vector = [0.1] * 384  # Replace with a real embedding vector
    results = milvus.query_embeddings(query_vector, top_k=5)

    # Display results
    print("Query Results:")
    for result in results[0]:
        print(f"Chunk ID: {result.id}, Distance: {result.distance}")
