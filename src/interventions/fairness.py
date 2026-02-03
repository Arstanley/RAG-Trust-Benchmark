from typing import List, Dict
import random
from collections import defaultdict

class FairnessRAG:
    """
    Implements Fair RAG (arXiv:2409.11598).
    Ensures fair exposure of different sources in the context.
    """
    def __init__(self, base_pipeline, top_k=5):
        self.base = base_pipeline
        self.top_k = top_k
        
    def _fair_rerank(self, docs: List[Dict]) -> List[Dict]:
        """
        Re-ranks docs to ensure diversity of sources.
        Algorithm: Group by source, then Round-Robin selection.
        """
        groups = defaultdict(list)
        for doc in docs:
            # Use 'source' or hash of content as group identifier
            group_id = doc.get("source", "unknown")
            if group_id == "unknown":
                group_id = hash(doc.get("content", "")[:50])
            groups[group_id].append(doc)
            
        # Round Robin
        reranked = []
        keys = list(groups.keys())
        random.shuffle(keys) # Randomize starting group
        
        while len(reranked) < self.top_k and any(groups.values()):
            for k in keys:
                if groups[k]:
                    reranked.append(groups[k].pop(0))
                if len(reranked) >= self.top_k:
                    break
                    
        return reranked

    def run(self, query: str) -> Dict:
        # 1. Retrieve MORE than needed (e.g. 2x)
        raw_context = self.base.retrieve(query, top_k=self.top_k * 3)
        
        # 2. Fair Rerank
        fair_context = self._fair_rerank(raw_context)
        
        # 3. Generate
        response = self.base.generate(query, fair_context)
        
        return {"response": response, "context": fair_context}
