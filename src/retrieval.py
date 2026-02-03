import time
from typing import List, Dict

class Retriever:
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        raise NotImplementedError

class WebRetriever(Retriever):
    def __init__(self, max_retries=3):
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.available = True
        except ImportError:
            print("⚠ duckduckgo-search not installed. Using MockRetriever.")
            self.available = False
            
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.available:
            return self._mock_retrieve(query, top_k)
            
        try:
            results = list(self.ddgs.text(query, max_results=top_k))
            # Normalize format
            docs = []
            for r in results:
                docs.append({
                    "content": r.get("body", "") or r.get("snippet", ""),
                    "title": r.get("title", ""),
                    "source": r.get("href", "")
                })
            return docs
        except Exception as e:
            print(f"⚠ Search failed: {e}")
            return self._mock_retrieve(query, top_k)

    def _mock_retrieve(self, query: str, top_k: int) -> List[Dict]:
        # Fallback for when internet is down or lib missing
        return [{
            "content": f"Simulated retrieval result {i} for query: {query}. This is a placeholder context containing relevant facts.",
            "title": f"Document {i}",
            "source": "simulated_kb"
        } for i in range(top_k)]
