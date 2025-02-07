import requests

class MilvusCloud:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint

    def create_collection(self, collection_name, dimension):
        """
        Create a collection in Milvus Cloud.
        :param collection_name: Collection name.
        :param dimension: Dimension of the vector embeddings (e.g., 384 for MiniLM).
        """
        url = f"{self.endpoint}/v2/vectordb/collections"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

        schema = {
            "collection_name": collection_name,
            "schema": {
                "fields": [
                    {
                        "name": "chunk_id",
                        "data_type": 5,  # DataType.INT64
                        "is_primary_key": True
                    },
                    {
                        "name": "embedding",
                        "data_type": 101,  # DataType.FLOAT_VECTOR
                        "dimension": dimension
                    }
                ]
            }
        }

        response = requests.post(url, headers=headers, json=schema)
        if response.status_code == 201:
            print(f"Collection '{collection_name}' created successfully!")
        else:
            print(f"Error creating collection: {response.status_code} - {response.json()}")

    def list_collections(self):
        """
        List all existing collections in Milvus Cloud.
        """
        url = f"{self.endpoint}/v2/vectordb/collections/list"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(url, headers=headers, json={})
        if response.status_code == 200:
            print("Collections in Milvus:")
            for collection in response.json()["data"]:
                print(f"- {collection}")
        else:
            print(f"Error listing collections: {response.status_code} - {response.json()}")


# Example Usage
if __name__ == "__main__":
    # Replace with your actual API key and public endpoint
    API_KEY = "f65478fc22acefefdb96955662fcbe5aa470a9d64e847d59b86a296f8fec1e54da5be7704857c10afcafe7a5eb9d106d947e6857"
    ENDPOINT = "https://in03-2c3b8bd41a90c08.serverless.gcp-us-west1.cloud.zilliz.com"

    milvus = MilvusCloud(api_key=API_KEY, endpoint=ENDPOINT)

    # Step 1: Create a new collection
    collection_name = "document_embeddings"
    vector_dimension = 384  # Replace with the dimension of your embeddings
    milvus.create_collection(collection_name, vector_dimension)

    # Step 2: List all collections
    milvus.list_collections()
