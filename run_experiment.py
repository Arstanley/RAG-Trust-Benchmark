import json
import os
from src.evaluator import Evaluator
from src.models import RAGPipeline

def main():
    print("Starting RAG Trustworthiness Benchmark...")
    
    # Define models
    models = {
        "Llama-3-RAG": RAGPipeline("Llama-3-8B-Instruct"),
        "Mistral-RAG": RAGPipeline("Mistral-7B-v0.3"),
        # Add a baseline for contrast
        "Naive-RAG": RAGPipeline("Small-Model-Baseline")
    }
    
    evaluator = Evaluator()
    results = {}
    
    # Run evaluation
    for name, model in models.items():
        # In a real experiment, we would generate responses here
        # responses = model.run_batch(dataset)
        scores = evaluator.run_full_eval(name)
        results[name] = scores
        
    # Save Results
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_data.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults saved. Now generating plots...")
    
    # Generate Visualization
    os.system("python3 src/visualize.py --results_file results/benchmark_data.json")

if __name__ == "__main__":
    main()
