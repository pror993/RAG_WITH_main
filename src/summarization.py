import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

class GeminiSummarizer:
    def __init__(self):
        """
        Initialize the Gemini API client with the API key.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)  # Initialize the Gemini client
        self.model = "gemini-2.0-flash"  # Use the recommended model for summarization

        if not self.api_key:
            raise ValueError("Gemini API key is not set. Please add it to the .env file.")

    def summarize(self, chunks, prompt="Summarize the following text:"):
        """
        Summarize the retrieved chunks using the Gemini API.
        :param chunks: List of retrieved text chunks.
        :param prompt: Instruction for summarization (optional).
        :return: Summarized response as a string.
        """
        # Combine the retrieved chunks into a single context
        context = " ".join([chunk["text"] for chunk in chunks])

        # Call the Gemini API to generate the summary
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=[f"{prompt}\n\n{context}"]
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API Error: {e}")


# Example Usage
if __name__ == "__main__":
    summarizer = GeminiSummarizer()

    # Example chunks
    chunks = [
        {"text": "Claim form and hospital bills are required."},
        {"text": "You need your ID proof and signed claim documents."},
        {"text": "Incomplete applications may cause delays."},
    ]

    # Summarize the chunks
    summary = summarizer.summarize(chunks)
    print("Summary:", summary)