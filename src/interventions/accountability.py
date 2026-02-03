import hashlib
import torch
from transformers import LogitsProcessor

class KGWWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab_size: int, gamma: float = 0.5, delta: float = 2.0, hash_key: int = 15485863):
        self.vocab_size = vocab_size
        self.gamma = gamma # Proportion of green list
        self.delta = delta # Logit bias magnitude
        self.hash_key = hash_key

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list:
        # Simple implementation: Hash the last token to seed RNG
        last_token = input_ids[-1].item()
        seed = (self.hash_key * last_token) % self.vocab_size
        rng = torch.Generator()
        rng.manual_seed(seed)
        
        perm = torch.randperm(self.vocab_size, generator=rng)
        green_size = int(self.vocab_size * self.gamma)
        return perm[:green_size]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            return scores
        green_list = self._get_greenlist_ids(input_ids[0])
        green_mask = torch.zeros_like(scores)
        green_mask[0, green_list] = 1 
        return scores + (green_mask * self.delta)

class AccountabilityRAG:
    """
    Implements Watermarking for Accountability.
    Note: Requires direct access to the model logits.
    """
    def __init__(self, base_pipeline, gamma=0.5, delta=2.0):
        self.base = base_pipeline
        # Access underlying HF model from StandardRAG -> HuggingFaceLLM
        if hasattr(self.base, 'llm') and hasattr(self.base.llm, 'tokenizer'):
            self.tokenizer = self.base.llm.tokenizer
            self.model = self.base.llm.model
            self.watermarker = KGWWatermarkLogitsProcessor(
                vocab_size=self.tokenizer.vocab_size,
                gamma=gamma,
                delta=delta
            )
        else:
            print("⚠ AccountabilityRAG requires a HuggingFaceLLM backend.")
            self.watermarker = None

    def generate(self, query, context):
        if not self.watermarker:
            return self.base.generate(query, context)
            
        # Replicate prompt formatting from StandardRAG
        context_str = "\n".join([f"[{doc['title']}] {doc['content']}" for doc in context])
        prompt = f"""Use the following context to answer the question.
        
Context:
{context_str}

Question: {query}

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            logits_processor=[self.watermarker],
            max_new_tokens=200,
            do_sample=True # Required for watermarking entropy
        )
        # Decode only the new tokens
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def run(self, query):
        context = self.base.retrieve(query)
        response = self.generate(query, context)
        return {"response": response, "context": context}
