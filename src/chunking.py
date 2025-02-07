import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from typing import List, Dict

class ChunkProcessor:
    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Initialize the chunk processor with chunk size and overlap.
        """
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from a PDF and split it into chunks. Pages with no extractable text are ignored.
        """
        reader = PdfReader(pdf_path)
        chunks = []
        for i, page in enumerate(reader.pages):
            # Extract text from the page
            page_text = page.extract_text()

            # Skip pages with no extractable text (likely images)
            if not page_text or page_text.strip() == "":
                print(f"Skipping page {i + 1}: No text found (likely contains only images).")
                continue

            # Split the text into chunks
            page_chunks = self.splitter.split_text(page_text)
            for chunk in page_chunks:
                chunks.append({"chunk": chunk, "pdf_name": os.path.basename(pdf_path), "page_number": i + 1})

        return chunks

    def save_chunks(self, chunks: List[Dict[str, str]], output_dir: str):
        """
        Save chunks to text files for embedding generation.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_dir, f"{chunks[i]['pdf_name']}_page_{chunks[i]['page_number']}_chunk_{i+1}.txt")
            with open(chunk_file, "w") as f:
                f.write(chunk["chunk"])
        print(f"Saved {len(chunks)} chunks to {output_dir}")

# Example Usage
if __name__ == "__main__":
    processor = ChunkProcessor(chunk_size=512, overlap=128)

    # Define input and output directories
    raw_docs_dir = "./raw_documents/"
    output_chunks_dir = "./processed_chunks/"

    # Process each PDF in the input folder
    for pdf_file in os.listdir(raw_docs_dir):
        if pdf_file.endswith(".pdf"):
            print(f"Processing file: {pdf_file}")
            chunks = processor.process_pdf(os.path.join(raw_docs_dir, pdf_file))
            processor.save_chunks(chunks, output_chunks_dir)
