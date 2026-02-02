import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import argparse

def plot_tradeoff_heatmap(data: pd.DataFrame, output_file="trust_tradeoff_heatmap.png"):
    """
    Generates a Heatmap of Impact Scores (Delta from Baseline).
    """
    plt.figure(figsize=(10, 6))
    
    # Calculate Deltas: Row - BaselineRow
    # Assuming 'Naive-RAG' is one of the rows
    baseline = data.loc['Naive-RAG']
    deltas = data.sub(baseline, axis=1)
    
    # Drop the baseline row from visualization (it would be all zeros)
    deltas = deltas.drop('Naive-RAG')

    sns.heatmap(deltas, annot=True, cmap="RdYlGn", center=0, fmt=".2f")
    plt.title("Impact of Trust Interventions (Delta vs. Naive Baseline)")
    plt.ylabel("Intervention Strategy")
    plt.xlabel("Trust Dimensions")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Trade-off Heatmap saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.results_file, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data).T 
    
    # Plot the raw scores
    plot_radar(df)
    
    # Plot the Trade-off Heatmap (New!)
    plot_tradeoff_heatmap(df)
