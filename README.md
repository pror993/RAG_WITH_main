# Retrieval-Augmented Generation (RAG) System for BPO Agents

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to assist BPO agents in answering complex queries by searching, reranking, and summarizing information from large collections of documents (such as insurance policies, claim forms, and guidelines). The system combines traditional information retrieval, state-of-the-art machine learning models, and LLM-based summarization to deliver concise, actionable answers.

---

## End-to-End Workflow

### 1. **Raw Document Upload**

- **Location:** Place your source PDF files in the `./raw_documents/` directory.
- **Purpose:** These are the documents from which the system will extract knowledge.

### 2. **Document Chunking (`src/chunking.py`)**

- **Process:**
  - Each PDF is read page by page.
  - Text is extracted from each page (pages with no text are skipped).
  - The text is split into overlapping chunks using LangChain’s `RecursiveCharacterTextSplitter` (default: 512 characters per chunk, 128 overlap).
  - Each chunk is saved as a `.txt` file in `./processed_chunks/`, named with the PDF, page, and chunk number.
- **Why:** Chunking enables efficient retrieval and ensures that answers are based on manageable, contextually relevant pieces of information.

### 3. **Embedding Generation (`src/embeddings.py`)**

- **Process:**
  - Each chunk file in `./processed_chunks/` is loaded.
  - The text is passed through a Hugging Face SentenceTransformer model (`all-MiniLM-L6-v2`) to generate a 384-dimensional vector embedding.
  - Each embedding is stored with its chunk filename in a list.
  - All embeddings are saved as a JSON file in `./embeddings/embeddings.json`.
- **Why:** Embeddings allow for semantic search, enabling the system to find text that is similar in meaning to the user’s query, not just matching keywords.

### 4. **Vector Database Indexing (`src/vector_db.py` or `src/create_collection.py`)**

- **Process:**
  - Connects to a Milvus Cloud instance using credentials from `.env`.
  - Creates a collection (table) for storing embeddings if it doesn’t exist.
  - Inserts all chunk embeddings from `embeddings.json` into the Milvus collection.
  - Creates an index for fast similarity search.
- **Why:** Milvus enables scalable, low-latency vector similarity search across all document chunks.

### 5. **Querying the System**

- **User Input:** The user submits a query (e.g., via API or CLI).
- **Query Embedding:** The query is embedded using the same SentenceTransformer model as the document chunks.
- **Pipeline Execution:** The `RAGPipeline` (`src/pipeline.py`) orchestrates the following steps:

#### a. **Hybrid Retrieval (`src/retrieval.py`)**

- **BM25 Search:** Uses TfidfVectorizer to find chunks with high keyword overlap with the query.
- **Vector Search:** Uses Milvus to find chunks whose embeddings are most similar to the query embedding.
- **Combining Results:** Both sets of results are returned for further processing, leveraging the strengths of both keyword and semantic search.

#### b. **Reranking (`src/reranking.py`)**

- **Process:** The top candidate chunks are passed to the `Reranker`, which uses a monoT5 transformer model to score each chunk for relevance to the query.
- **Result:** Chunks are sorted by their relevance score, ensuring the most relevant information is prioritized.

#### c. **Summarization (`src/summarization.py`)**

- **Process:** The most relevant chunks are passed to the `GeminiSummarizer`.
  - Chunks are concatenated into a single context.
  - A structured prompt is created, instructing Gemini to focus on actionable, customer-facing details.
  - The Gemini API is called with this prompt and context.
  - The API returns a concise, helpful summary.
- **Result:** The summary is returned to the user.

---

## System Architecture

```mermaid
graph TD
    A[Raw PDFs in raw_documents/] --> B[Chunking (chunking.py)]
    B --> C[Text Chunks in processed_chunks/]
    C --> D[Embedding Generation (embeddings.py)]
    D --> E[embeddings.json]
    E --> F[Milvus Vector DB (vector_db.py)]
    G[User Query] --> H[Query Embedding]
    H --> I[RAGPipeline (pipeline.py)]
    F --> I
    I --> J[Hybrid Retrieval (BM25 + Vector)]
    J --> K[Reranking (monoT5)]
    K --> L[Summarization (Gemini API)]
    L --> M[Final Answer]
```

---

## Key Components and Their Roles

### - **ChunkProcessor (`src/chunking.py`):**
  - Splits PDFs into overlapping, context-preserving text chunks.
  - Handles extraction and storage of chunked text.

### - **EmbeddingGenerator (`src/embeddings.py`):**
  - Uses Hugging Face SentenceTransformer to convert text chunks into dense vector embeddings.
  - Stores embeddings for later retrieval.

### - **MilvusDB (`src/vector_db.py`):**
  - Manages connection to Milvus vector database.
  - Handles collection creation, indexing, and insertion of embeddings.
  - Supports fast vector similarity search.

### - **HybridRetriever (`src/retrieval.py`):**
  - Performs both BM25 (keyword-based) and vector (semantic) search.
  - Combines results for robust retrieval.

### - **Reranker (`src/reranking.py`):**
  - Uses monoT5 transformer model to rerank candidate chunks by relevance to the query.
  - Ensures the most contextually appropriate information is surfaced.

### - **GeminiSummarizer (`src/summarization.py`):**
  - Calls the Gemini API with a structured prompt and the most relevant chunks.
  - Produces a concise, actionable summary tailored for BPO agents and customers.

### - **RAGPipeline (`src/pipeline.py`):**
  - Orchestrates the entire process: retrieval, reranking, and summarization.
  - Provides a single entry point for answering user queries.

---

## Example End-to-End Usage

1. **Upload PDFs:** Place files in `./raw_documents/`.
2. **Chunk Documents:**  
    ```sh
    python src/chunking.py
    ```
3. **Generate Embeddings:**  
    ```sh
    python src/embeddings.py
    ```
4. **Index Embeddings in Milvus:**  
    ```sh
    python src/vector_db.py
    ```
5. **Start the API Server:**  
    ```sh
    uvicorn src.api:app --reload
    ```
6. **Query the System:**  
    Send a POST request to `/query` endpoint:
    ```json
    {
        "query": "What documents are required for a health insurance claim?",
        "top_k": 5
    }
    ```
    **Sample Response:**
    ```json
    {
        "summary": "To file a health insurance claim, you need to provide the following documents: a completed claim form, a copy of your insurance card, medical reports, and receipts for medical expenses. Ensure all documents are accurate and complete to avoid delays in processing."
    }
    ```

---

## Design Choices and Rationale

- **Chunking:** Enables fine-grained retrieval and avoids context loss from large documents.
- **Hybrid Retrieval:** Combines the precision of keyword search (BM25) with the flexibility of semantic search (vector embeddings).
- **Reranking:** monoT5 leverages deep language understanding to prioritize the most relevant chunks.
- **Summarization:** Gemini API produces user-friendly, actionable summaries, guided by structured prompts.
- **Extensibility:** New documents can be added by repeating the chunking, embedding, and indexing steps.

---

## Tech Stack

- **Python**: Core language for all modules.
- **FastAPI**: High-performance API framework.
- **Pydantic**: Data validation and settings management.
- **Transformers (Hugging Face)**: For embedding and reranking models.
- **Milvus**: Scalable vector database for fast similarity search.
- **Google Gemini API**: For high-quality LLM-based summarization.
- **dotenv**: Secure management of environment variables.
- **LangChain**: Advanced text chunking.
- **PyPDF2**: PDF text extraction.
- **Logging**: For robust error tracking and debugging.

---

## Environment Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/RAG_WITH_main.git
    cd RAG_WITH_main
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory with:
        ```
        GEMINI_API_KEY=your_gemini_api_key
        MILVUS_PUBLIC_ENDPOINT=your_milvus_endpoint
        MILVUS_API_KEY=your_milvus_api_key
        ```

---

## Advanced Topics

- **Adding New Documents:**  
  Place new PDFs in `raw_documents/` and rerun the chunking, embedding, and indexing scripts.
- **Model Customization:**  
  Swap out embedding or reranking models in `embeddings.py` or `reranking.py` for domain-specific needs.
- **Prompt Engineering:**  
  Modify the prompt in `summarization.py` to tailor summaries for different use cases.
- **Scaling:**  
  Milvus and FastAPI can be horizontally scaled for large document collections and high query volumes.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For questions or support, please open an issue on GitHub.