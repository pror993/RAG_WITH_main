import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict


class Reranker:
    def __init__(self, model_name="castorini/monot5-base-msmarco"):
        """
        Initialize the monoT5 reranker with the specified model.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info(f"Loading reranker model: {model_name}")

        # Use the slow tokenizer to avoid conversion issues
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def format_to_t5_query(self, query: str, candidate: str) -> str:
        """
        Format query-candidate pair for monoT5 input.
        """
        return f"Query: {query} Document: {candidate} Relevant:"

    def rerank(self, query: str, candidates: List[Dict[str, str]]) -> List[Dict[str, float]]:
        """
        Rerank candidates based on relevance to the query.
        :param query: The query text.
        :param candidates: A list of candidate documents (chunks).
        :return: A list of candidates with relevance scores.
        """
        reranked_results = []

        for candidate in candidates:
            try:
                # Format input for monoT5
                t5_input = self.format_to_t5_query(query, candidate["text"])
                inputs = self.tokenizer(t5_input, return_tensors="pt", max_length=512, truncation=True)

                # Generate textual classification (e.g., "true"/"false")
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=1)
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

                # Map "true"/"false" to numeric scores
                if prediction == "true":
                    relevance_score = 1.0
                elif prediction == "false":
                    relevance_score = 0.0
                else:
                    relevance_score = 0.5  # Default to a neutral score for unexpected outputs

                # Append result with relevance score
                reranked_results.append({**candidate, "relevance_score": relevance_score})

            except Exception as e:
                self.logger.error(f"Error processing candidate: {e}")
                reranked_results.append({**candidate, "relevance_score": 0.5})  # Default to neutral score on error

        # Sort candidates by relevance score
        reranked_results = sorted(reranked_results, key=lambda x: x["relevance_score"], reverse=True)
        return reranked_results
