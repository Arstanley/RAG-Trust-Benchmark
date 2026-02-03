from typing import List, Dict

class RobustnessRAG:
    """
    Implements 'Astute RAG' (arXiv:2410.07176).
    Consolidates Internal Knowledge (LLM param) and External Knowledge (Retrieval).
    """
    def __init__(self, base_pipeline):
        self.base = base_pipeline
        
    def generate(self, query: str, context: List[Dict]) -> str:
        # 1. Elicit Internal Knowledge
        internal_prompt = f"Answer the following question using only your internal knowledge.\nQuestion: {query}\nAnswer:"
        internal_knowledge = self.base.llm.generate(internal_prompt)
        
        # 2. Get External Knowledge (Standard Generation)
        # Note: context is already retrieved by the calling .run() method usually, 
        # but here we generate the answer based on it.
        external_ans = self.base.generate(query, context)
        
        # 3. Consolidate and Resolve Conflicts
        consolidation_prompt = f"""You are an expert at resolving conflicts between internal knowledge and retrieved information.
        
Question: {query}

Internal Knowledge:
{internal_knowledge}

External Retrieved Knowledge:
{external_ans}

Task: Consolidate the answer. If they conflict, prioritize the External Knowledge only if it seems more specific and reliable. If the External Knowledge looks like a hallucination or irrelevant, trust Internal Knowledge.

Final Answer:"""
        
        final_response = self.base.llm.generate(consolidation_prompt)
        return final_response

    def run(self, query: str) -> Dict:
        context = self.base.retrieve(query)
        response = self.generate(query, context)
        return {"response": response, "context": context}
