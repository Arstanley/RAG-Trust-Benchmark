from typing import List, Dict, Any
from .base import RAGPipeline
from ..models import HuggingFaceLLM
from ..retrieval import WebRetriever

class StandardRAG(RAGPipeline):
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.llm = HuggingFaceLLM(model_name)
        self.retriever = WebRetriever()
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.retriever.retrieve(query, top_k)

    def generate(self, query: str, context: List[Dict]) -> str:
        context_str = "\n".join([f"[{doc['title']}] {doc['content']}" for doc in context])
        prompt = f"""Use the following context to answer the question.
        
Context:
{context_str}

Question: {query}

Answer:"""
        return self.llm.generate(prompt)
        
    def run(self, query: str) -> Dict[str, Any]:
        context = self.retrieve(query)
        response = self.generate(query, context)
        return {"response": response, "context": context}
