import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class GeminiSummarizer:
    def __init__(self):
        """
        Initialize the Gemini API summarizer with the API key.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.endpoint = "https://api.gemini.com/v1/summarize"  # Replace with the actual Gemini API endpoint for summarization

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

        # Prepare the payload for the API request
        payload = {
            "prompt": f"{prompt}\n\n{context}",
            "max_tokens": 200,  # Limit the length of the summary
            "temperature": 0.7,  # Adjust for more/less creativity
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Call the Gemini API
        response = requests.post(self.endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")

        return response.json()["summary"]  # Adjust the key based on Gemini's API response format


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
