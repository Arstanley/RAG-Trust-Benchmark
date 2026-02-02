import numpy as np
from typing import List, Dict

class FairnessRAG:
    """
    Implements Fair Reranking to ensure diverse representation in context.
    Algorithm: Maximal Marginal Relevance (MMR)
    """
    def __init__(self, base_pipeline, lambda_param: float = 0.5):
        self.base = base_pipeline
        self.lambda_param = lambda_param

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def mmr_rerank(self, query_emb, doc_embs, docs, top_k=5):
        """
        Selects documents that are relevant to query but distinct from already selected docs.
        """
        selected_indices = []
        candidate_indices = list(range(len(docs)))

        while len(selected_indices) < top_k and candidate_indices:
            best_score = -float('inf')
            best_idx = -1

            for idx in candidate_indices:
                # Relevance score
                rel_score = self._cosine_sim(query_emb, doc_embs[idx])
                
                # Diversity score (max sim to already selected)
                if not selected_indices:
                    div_score = 0
                else:
                    div_score = max([self._cosine_sim(doc_embs[idx], doc_embs[sel_idx]) 
                                   for sel_idx in selected_indices])
                
                # MMR Equation
                mmr_score = (self.lambda_param * rel_score) - ((1 - self.lambda_param) * div_score)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
        return [docs[i] for i in selected_indices]

    def generate(self, query: str, context: List[Dict]) -> str:
        # 1. Embed Query and Docs (Mock call)
        # q_emb = model.encode(query)
        # d_embs = model.encode([c['text'] for c in context])
        
        # 2. Rerank for Fairness/Diversity
        # reranked_context = self.mmr_rerank(q_emb, d_embs, context)
        
        # Mock passthrough
        reranked_context = context # Placeholder without actual embeddings
        
        return self.base.generate(query, reranked_context)
