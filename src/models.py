import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class HuggingFaceLLM(LLM):
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        print(f"Loading {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Use auto device map (GPU if available)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
            self.available = True
        except Exception as e:
            print(f"⚠ Could not load model {model_name}: {e}")
            print("  Falling back to MockLLM for testing.")
            self.available = False
            self.mock = MockLLM(model_name)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        if not self.available:
            return self.mock.generate(prompt)
        
        try:
            messages = [
                {"role": "user", "content": prompt},
            ]
            outputs = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            # Fallback for older transformers versions or raw text models
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class MockLLM(LLM):
    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str) -> str:
        return f"Response from {self.name} (Mock) based on prompt length {len(prompt)}."
