from collections import Counter
from typing import List, Dict

class ReliabilityRAG:
    """
    Implements Self-Consistency (Majoriy Voting) to improve robustness.
    """
    def __init__(self, base_pipeline, n_paths: int = 3):
        self.base = base_pipeline
        self.n_paths = n_paths

    def generate(self, query: str, context: List[Dict]) -> str:
        candidates = []
        
        # 1. Generate N paths (with high temperature usually)
        for _ in range(self.n_paths):
            # In production, set do_sample=True, temperature=0.7
            response = self.base.generate(query, context)
            candidates.append(response)
            
        # 2. Majority Vote (Simplistic string matching)
        # Real impl needs semantic equivalence check (e.g. using NLI)
        most_common = Counter(candidates).most_common(1)[0][0]
        
        return most_common
