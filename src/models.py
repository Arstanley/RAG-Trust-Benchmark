import random
from typing import List, Dict

class MockRAGModel:
    def __init__(self, name: str):
        self.name = name
        print(f"Loading {name}...")

    def generate(self, query: str, context: List[str]) -> str:
        """
        Mock generation. In a real experiment, this calls HuggingFace or OpenAI.
        """
        return f"Response from {self.name} based on {len(context)} docs."

class RAGPipeline:
    def __init__(self, model_name: str, retriever_name: str = "faiss"):
        self.generator = MockRAGModel(model_name)
        self.retriever_name = retriever_name

    def run(self, query: str) -> str:
        # Mock retrieval
        docs = ["Doc 1 content...", "Doc 2 content..."]
        response = self.generator.generate(query, docs)
        return response
