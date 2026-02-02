# RAG Trustworthiness Benchmark: Trade-off Analysis

This repository contains the experimental framework to benchmark Retrieval-Augmented Generation (RAG) systems across six key dimensions of trustworthiness. The goal is to visualize the trade-offs between these dimensions (e.g., does increasing Safety reduce Truthfulness?).

## 🎯 Objectives
1.  Evaluate two backbone RAG models (e.g., Llama-3 vs. Mistral) on 6 trust dimensions.
2.  Analyze the correlation and trade-offs between dimensions.
3.  Produce a **Trust Trade-off Heatmap** and Radar Charts.

## 📊 The 6 Trust Dimensions
We define the following dimensions for evaluation:
1.  **Truthfulness**: Faithfulness to the retrieved context and factual accuracy (avoiding hallucinations).
2.  **Safety**: Resistance to jailbreaks, toxic queries, and harmful content generation.
3.  **Robustness**: Stability of performance against retrieval noise (irrelevant documents) and prompt perturbations.
4.  **Fairness**: Performance consistency across different demographics/dialects to avoid bias.
5.  **Privacy**: Ability to detect and redact Personally Identifiable Information (PII) and refuse private data extraction.
6.  **Transparency** (or Citability): Accuracy of citations and source attribution.

## 🧪 Experiment Design

### 1. Models (Backbones)
- **Model A**: `meta-llama/Meta-Llama-3-8B-Instruct` (High capability)
- **Model B**: `mistralai/Mistral-7B-Instruct-v0.3` (Efficient, high performance)
*Both will use the same retriever (e.g., FAISS + Contriever).*

### 2. Datasets & Metrics
| Dimension | Dataset Source | Metric |
|-----------|----------------|--------|
| **Truthfulness** | TruthfulQA / RAGAS | Faithfulness Score (0-1) |
| **Safety** | Do-Not-Answer / SafetyBench | Refusal Rate / Harmlessness Score |
| **Robustness** | RGB (Retrieval Augmented Generation Benchmark) | Noise Tolerance (Acc drop %) |
| **Fairness** | BBQ (Bias Benchmark for QA) | Bias Score (lower is better) |
| **Privacy** | EnterprisePII / Confident-PII | PII Leakage Rate |
| **Transparency** | ALCE / CitationBench | Citation Precision/Recall |

### 3. Visualization
- **Heatmap**: Correlation matrix of scores across dimensions (to find negative correlations).
- **Radar Chart**: Visual comparison of Model A vs. Model B area.

## 🚀 Usage

### Installation
```bash
pip install -r requirements.txt
```

### Download Datasets
The benchmark automatically downloads datasets on first run. To pre-fetch:
```bash
python src/data/loader.py
```

This will download:
- **AdvBench** (520 harmful behaviors for safety testing)
- **BBQ** (Bias Benchmark across 9 categories: Age, Disability, Gender, Nationality, Physical Appearance, Race/Ethnicity, Religion, Socioeconomic Status, Sexual Orientation)
- **Synthetic TruthfulQA** (50 factual questions)
- **Synthetic PII prompts** (50 privacy-testing examples)
- **RGB robustness tests** (50 perturbed prompts)

**Dataset Summary:**
- Safety: 50 samples from AdvBench
- Fairness: 45 samples across BBQ categories
- Reliability: 50 synthetic truth-testing questions
- Privacy: 50 PII-containing prompts
- Robustness: 50 typo-perturbed questions

### Run Evaluation
```bash
python run_experiment.py --models llama3,mistral --dimensions all
```

### Generate Plots
```bash
python src/visualize.py --results_file results/benchmark_data.json
```
