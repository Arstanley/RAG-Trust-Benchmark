from abc import ABC, abstractmethod
from typing import List, Dict, Any

class RAGPipeline(ABC):
    def __init__(self, model_path: str, retriever: Any):
        self.model_path = model_path
        self.retriever = retriever
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents."""
        pass

    @abstractmethod
    def generate(self, query: str, context: List[Dict]) -> str:
        """Generate a response given query and context."""
        pass

    def run(self, query: str) -> Dict[str, Any]:
        """End-to-end execution."""
        context = self.retrieve(query)
        response = self.generate(query, context)
        return {
            "query": query,
            "context": context,
            "response": response
        }
