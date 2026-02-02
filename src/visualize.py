import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import argparse

def plot_heatmap(data: pd.DataFrame, output_file="trust_tradeoff_heatmap.png"):
    """
    Generates a correlation heatmap between dimensions.
    """
    plt.figure(figsize=(10, 8))
    # For a tradeoff heatmap, we might look at correlation across many runs,
    # OR simply visualize the scores of multiple models side-by-side.
    # Let's plot the raw scores as a heatmap (Models vs Dimensions).
    
    sns.heatmap(data, annot=True, cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("RAG Trustworthiness Dimensions Heatmap")
    plt.ylabel("Models")
    plt.xlabel("Trust Dimensions")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")

def plot_radar(data: pd.DataFrame, output_file="trust_radar.png"):
    """
    Generates a Radar Chart comparing models.
    """
    labels = list(data.columns)
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for model_name, row in data.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Trust Trade-off Radar")
    plt.savefig(output_file)
    print(f"Radar chart saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.results_file, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data).T # Transpose so Models are rows
    
    plot_heatmap(df)
    plot_radar(df)
