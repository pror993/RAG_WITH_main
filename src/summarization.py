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

    def summarize(self, chunks, query: str, prompt="Summarize the following text:"):
        """
        Summarize the retrieved chunks using the Gemini API.
        :param chunks: List of retrieved text chunks.
        :param query: The original user query.
        :param prompt: Instruction for summarization (optional).
        :return: Summarized response as a string.
        """
        # Combine the retrieved chunks into a single context
        context = " ".join([chunk["text"] for chunk in chunks])

        # Default structured prompt for assisting a BPO agent
        default_prompt = (
            "You are assisting a BPO agent by summarizing the following claims document. "
            "Focus on key details such as required documents, conditions, important notes, and any actions the customer needs to take. "
            "Ensure the summary is accurate, easy to understand, and helpful for the customer. "
            "If the source of the information is mentioned in the context, include it in the summary."
        )
        
        final_prompt = f"{default_prompt}\n\nQuery: {query}\n\n{prompt}\n\n{context}"

        # Call the Gemini API to generate the summary
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=[final_prompt]
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API Error: {e}")
