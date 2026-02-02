import json
import os
from src.evaluator import Evaluator
from src.models import RAGPipeline

def main():
    print("Starting RAG Trustworthiness Benchmark...")
    
    # Run evaluation for Llama-3-8B variants
    backbone = "Llama-3-8B-Instruct"
    variants = [
        "Naive-RAG",
        "Safety-RAG",
        "Privacy-RAG",
        "Accountability-RAG",
        "Fairness-RAG"
    ]
    
    evaluator = Evaluator()
    results = {}
    
    # Run evaluation
    for variant in variants:
        # In real experiment, we'd init the specific pipeline here:
        # pipeline = RAGPipeline(backbone, intervention=variant.split("-")[0])
        # responses = pipeline.run_batch(...)
        
        scores = evaluator.run_full_eval(variant, backbone)
        results[variant] = scores
        
    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_data.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults saved. Now generating plots...")
    
    # Generate Visualization
    os.system("python3 src/visualize.py --results_file results/benchmark_data.json")

if __name__ == "__main__":
    main()
