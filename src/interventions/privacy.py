from typing import List, Dict
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    AnalyzerEngine = None
    AnonymizerEngine = None

class PrivacyRAG:
    """
    Implements Privacy-Preserving RAG (arXiv:2402.16893 style defense).
    Scrubs PII from Query before retrieval, and from Context before generation.
    """
    def __init__(self, base_pipeline):
        self.base = base_pipeline
        if AnalyzerEngine:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            print("⚠ Presidio not installed. Privacy scrubbing disabled.")
            self.analyzer = None

    def _scrub(self, text: str) -> str:
        if not self.analyzer or not text:
            return text
        results = self.analyzer.analyze(text=text, entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"], language='en')
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text

    def run(self, query: str) -> Dict:
        # 1. Scrub Query
        clean_query = self._scrub(query)
        
        # 2. Retrieve using clean query (prevents leaking PII to retrieval system/logs)
        context = self.base.retrieve(clean_query)
        
        # 3. Scrub Context (prevent leaking PII from docs to LLM context window)
        clean_context = []
        for doc in context:
            clean_context.append({
                "title": doc.get("title", ""),
                "content": self._scrub(doc.get("content", "")),
                "source": doc.get("source", "")
            })
            
        # 4. Generate
        response = self.base.generate(clean_query, clean_context)
        
        return {"response": response, "context": clean_context}
