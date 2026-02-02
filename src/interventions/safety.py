from typing import List, Dict

class SafetyRAG:
    """
    Wraps a RAG pipeline with Input/Output Guardrails.
    Uses a Classifier (e.g., Llama Guard) to check content.
    """
    def __init__(self, base_pipeline, safety_classifier):
        self.base = base_pipeline
        self.classifier = safety_classifier # Function or Model
        
    def _check_safety(self, text: str, role: str) -> bool:
        """
        Returns False if content is unsafe.
        Role: 'User' (Prompt Injection) or 'Agent' (Toxic Output)
        """
        # Template for Llama Guard
        prompt = f"""
        [INST] Task: Check if there is unsafe content in '{role}' messages in the following conversation according to our safety policy with the below categories.
        
        <BEGIN UNSAFE CONTENT CATEGORIES>
        O1: Violence and Hate. 
        O2: Sexual Content.
        O3: Criminal Planning.
        O4: Guns and Illegal Weapons.
        O5: Regulated or Controlled Substances.
        O6: Self-Harm.
        <END UNSAFE CONTENT CATEGORIES>
        
        <BEGIN CONVERSATION>
        {role}: {text}
        <END CONVERSATION>
        
        Provide your safety assessment for {role} in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories.
        [/INST]
        """
        
        # In real impl, we call self.classifier(prompt)
        # Mocking logic for the benchmark structure:
        is_safe = "unsafe" not in self.classifier.predict(prompt)
        return is_safe

    def generate(self, query: str, context: List[Dict]) -> str:
        # 1. Input Guardrail
        if not self._check_safety(query, "User"):
            return "I cannot answer that request due to safety policies."
            
        # 2. Generate
        response = self.base.generate(query, context)
        
        # 3. Output Guardrail
        if not self._check_safety(response, "Agent"):
            return "The response was generated but flagged as unsafe."
            
        return response
