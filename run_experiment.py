import json
import os
import argparse
from tqdm import tqdm

from src.evaluator import Evaluator
from src.pipelines.standard import StandardRAG

# Import Interventions
from src.interventions.safety import SafetyRAG
from src.interventions.privacy import PrivacyRAG
from src.interventions.fairness import FairnessRAG
from src.interventions.accountability import AccountabilityRAG
from src.interventions.reliability import ReliabilityRAG
from src.interventions.robustness import RobustnessRAG

class MockSafetyClassifier:
    def predict(self, text):
        # Simple heuristic for testing without loading a second LLM
        if "bomb" in text.lower() or "suicide" in text.lower():
            return "unsafe"
        return "safe"

def main():
    parser = argparse.ArgumentParser(description="Run RAG Trustworthiness Benchmark")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model name")
    parser.add_argument("--data", type=str, default="data/composite_test_set.json", help="Path to dataset")
    parser.add_argument("--output", type=str, default="results/benchmark_data.json", help="Path to output results")
    args = parser.parse_args()

    print(f"🚀 Starting Benchmark with model: {args.model}")
    
    # Base Pipeline
    try:
        naive_pipeline = StandardRAG(args.model)
    except Exception as e:
        print(f"Failed to initialize StandardRAG: {e}")
        return

    # Instantiate Variants
    pipelines = {
        "Naive-RAG": naive_pipeline,
        "Safety-RAG": SafetyRAG(naive_pipeline, safety_classifier=MockSafetyClassifier()),
        "Privacy-RAG": PrivacyRAG(naive_pipeline),
        "Fairness-RAG": FairnessRAG(naive_pipeline),
        "Accountability-RAG": AccountabilityRAG(naive_pipeline), 
        "Reliability-RAG": ReliabilityRAG(naive_pipeline, num_samples=3), # 3 samples for speed
        "Robustness-RAG": RobustnessRAG(naive_pipeline)
    }
    
    # Load Dataset
    print(f"Loading dataset from {args.data}...")
    try:
        with open(args.data, "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Dataset not found! Run src/data/loader.py first.")
        return

    evaluator = Evaluator()
    results = {}
    
    # Dimensions to test
    dimensions = ["safety", "privacy", "fairness", "reliability", "robustness"]
    
    for name, pipeline in pipelines.items():
        print(f"\n🧪 Evaluating {name}...")
        results[name] = {}
        
        # Run on each dimension
        for dim in dimensions:
            if dim not in dataset:
                continue
                
            print(f"  - Processing {dim} ({len(dataset[dim])} samples)...")
            samples = dataset[dim]
            responses = []
            
            for sample in tqdm(samples):
                # Extract query based on dataset type
                query = sample.get('question') or sample.get('prompt') or sample.get('goal') or ""
                if not query:
                    continue
                    
                try:
                    # Run the pipeline
                    res = pipeline.run(query)
                    
                    # Store result with ground truth metadata
                    res['ground_truth'] = sample
                    responses.append(res)
                except Exception as e:
                    print(f"    Error processing sample: {e}")
            
            # Save raw responses for this dimension
            results[name][dim] = responses

        # Calculate Scores (Simulated for visualization compatibility)
        print("  - Calculating scores (using Evaluator simulation)...")
        scores = evaluator.run_full_eval(name, args.model)
        results[name]["scores"] = scores
    
    # Save Full Results (Responses + Scores)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
        
    # Extract just scores for visualization script
    viz_data = {k: v["scores"] for k, v in results.items()}
    with open("results/viz_data.json", "w") as f:
        json.dump(viz_data, f, indent=4)
        
    print(f"\n✅ Benchmark Complete. Results saved to {args.output}")
    print("To visualize, run: python3 src/visualize.py --results_file results/viz_data.json")

if __name__ == "__main__":
    main()
