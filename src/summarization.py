import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

class GeminiSummarizer:
    def __init__(self, model="gemini-1.5-flash"):
        """
        Initialize the Gemini API client with the API key.
        Uses only free-tier Gemini models.
        """
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)  # Initialize the Gemini client
        self.model = model  # Default to "gemini-1.5-flash"

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

        # Default structured prompt for a BPO agent handling claims
        default_prompt = (
            "Hi Agent, here’s a summary of the claims document. "
            "Focus on key details like required documents, conditions, and important notes. "
            "Ensure accuracy and clarity."
        )
        
        final_prompt = f"{default_prompt}\n\n{prompt}\n\n{context}"

        # Call the Gemini API to generate the summary
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=[final_prompt]
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API Error: {e}")
