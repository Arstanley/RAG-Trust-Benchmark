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
        # KGW Algorithm: Split vocab into Green/Red based on previous token
        # Boost Green tokens by delta
        
        # Note: In production, batch processing is needed. simplifying for clarity.
        if input_ids.shape[1] == 0:
            return scores

        green_list = self._get_greenlist_ids(input_ids[0]) # Use first seq in batch
        
        # Create a mask for green list tokens
        green_mask = torch.zeros_like(scores)
        green_mask[0, green_list] = 1 
        
        return scores + (green_mask * self.delta)

class AccountabilityRAG:
    def __init__(self, base_pipeline, gamma=0.5, delta=2.0):
        self.base = base_pipeline
        self.watermarker = KGWWatermarkLogitsProcessor(
            vocab_size=self.base.tokenizer.vocab_size,
            gamma=gamma,
            delta=delta
        )
    
    def generate(self, query, context):
        # Inject the logits processor into the generation call
        input_text = self.base.format_prompt(query, context)
        inputs = self.base.tokenizer(input_text, return_tensors="pt")
        
        outputs = self.base.model.generate(
            **inputs,
            logits_processor=[self.watermarker],
            max_new_tokens=200
        )
        return self.base.tokenizer.decode(outputs[0])
