# Retrieval-Augmented Generation (RAG) Application for BPO Agents

## Overview

This project is a Retrieval-Augmented Generation (RAG) application designed to assist BPO agents by retrieving, reranking, and summarizing relevant information from a large document collection. The application uses a combination of traditional information retrieval techniques and modern machine learning models to provide concise and accurate summaries in response to user queries.

## Key Components

1. **User Query Input**:
    - The user submits a query via a FastAPI endpoint.
    - The query is received by the `query_rag` function in `src/api.py`.

2. **Query Embedding**:
    - A placeholder query embedding is generated (this should be replaced with actual embedding logic).

3. **RAG Pipeline Execution**:
    - The `RAGPipeline` class in `src/pipeline.py` orchestrates the retrieval, reranking, and summarization processes.
    - The `run` method of the `RAGPipeline` class is called with the user query, query embedding, and the number of top results to retrieve (`top_k`).

4. **Retrieval**:
    - The `HybridRetriever` class in `src/retrieval.py` retrieves relevant document chunks using both BM25 (a traditional information retrieval algorithm) and vector-based search.
    - The retrieved results are combined and prepared for reranking.

5. **Reranking**:
    - The `Reranker` class in `src/reranking.py` reranks the retrieved document chunks based on their relevance to the query using a monoT5 model.
    - The reranked results are sorted by relevance score.

6. **Summarization**:
    - The `GeminiSummarizer` class in `src/summarization.py` summarizes the top reranked document chunks using the Gemini API.
    - The summarization prompt includes the original user query and a structured prompt to guide the summarization process.

7. **Response Generation**:
    - The final summarized response is returned to the user via the FastAPI endpoint.

## Innovative Ideas

- **Hybrid Retrieval**: Combining BM25 and vector-based search to leverage the strengths of both traditional and modern retrieval methods.
- **Reranking with monoT5**: Using a state-of-the-art transformer model to rerank retrieved document chunks based on their relevance to the query.
- **Summarization with Gemini API**: Utilizing an external API to generate high-quality summaries, ensuring that the final output is concise and informative.
- **Structured Prompts**: Providing detailed and structured prompts to guide the summarization process, ensuring that the summaries are tailored to the needs of BPO agents.

## Tech Stack

- **Python**: The primary programming language used for the application.
- **FastAPI**: A modern web framework for building APIs with Python.
- **Pydantic**: Used for data validation and settings management.
- **Transformers**: A library by Hugging Face for working with transformer models like monoT5.
- **Milvus**: A vector database used for storing and querying embeddings.
- **Google Gemini API**: An external API used for generating summaries.
- **dotenv**: A library for loading environment variables from a `.env` file.
- **Logging**: Used for logging important events and errors.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/RAG_WITH_main.git
    cd RAG_WITH_main
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a [.env](http://_vscodecontentref_/0) file in the root directory and add the necessary environment variables as shown in [.env](http://_vscodecontentref_/1).

## Usage

1. Start the FastAPI server:
    ```sh
    uvicorn src.api:app --reload
    ```

2. Send a POST request to the `/query` endpoint with a JSON payload containing the query and [top_k](http://_vscodecontentref_/2) (optional):
    ```json
    {
        "query": "What documents are required for a health insurance claim?",
        "top_k": 5
    }
    ```

3. The server will return a JSON response with the summarized answer.

## Example

Here is an example of how the application works:

1. **User Query**:
    - The user submits a query: "What documents are required for a health insurance claim?"

2. **Pipeline Execution**:
    - The `RAGPipeline` retrieves relevant document chunks, reranks them, and summarizes the top results.

3. **Final Summary**:
    - The application returns a summary: "To file a health insurance claim, you need to provide the following documents: a completed claim form, a copy of your insurance card, medical reports, and receipts for medical expenses. Ensure all documents are accurate and complete to avoid delays in processing."

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.