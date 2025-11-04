# 🧠 Causal-CoT: Causal Chain-of-Thought for Validated Reasoning

> Official implementation of **Causal-CoT**, a framework that integrates causal graph construction, augmentation, and verification into the Chain-of-Thought (CoT) paradigm.  
> Paper: *Causal-CoT: Causal Chain-of-Thought for Validated Reasoning* (under review at ICLR 2026)

---

## 📜 Overview

Chain-of-Thought (CoT) prompting enables large language models (LLMs) to produce step-by-step reasoning. However, generated rationales are often **unfaithful** or **logically inconsistent**.  
**Causal-CoT** mitigates these problems by turning linear CoT traces into structured, verifiable causal graphs through a three-stage pipeline:

1. **DAG-guided CoT** — construct an initial Directed Acyclic Graph (DAG) from the premise and hypothesis.  
2. **Reflection & Augmentation** — enrich the DAG by prompting for missing mediators, confounders, or contextual variables.  
3. **Causal Verification** — estimate conditional probabilities (via LLM prompts), apply do-calculus, and verify causal effects quantitatively.

This pipeline improves reasoning **fidelity**, **interpretability**, and **stability** across mathematics, commonsense, and causal reasoning benchmarks.

---

## 🧩 Key Features

- **Graph-structured reasoning**: map CoT steps to DAG nodes/edges.  
- **Do-calculus verification**: quantitative causal effect estimation and hypothesis acceptance.  
- **Prompt-based augmentation**: elicit missing premises, mediators, or confounders from the LLM.  
- **IR backends (optional)**: integrate web / knowledge graph / RAG evidence for uncertain edges.  
- **Modular & extensible**: clear separation of DAG construction, augmentation, and verification.

---

## 🔖 Repository snapshot

```
Causal-CoT/
├── causal_cot/                # Core pipeline implementation
│   ├── stable_run.py          # Entry script for experiments
│   ├── dag_construction.py    # Stage I: DAG-guided CoT
│   ├── augmentation.py        # Stage II: Reflection & Augmentation
│   └── verification.py        # Stage III: Causal Verification
├── causalnet/                 # Example benchmark datasets (CSV)
├── docs/                      # Documentation & figures (e.g., pipeline image)
├── requirements.txt
├── LICENSE
└── README.md
```

> **Note:** actual filenames inside `causal_cot/` may vary slightly depending on the implementation; the above shows the recommended logical layout.

---

## ⚙️ Quick start (exact steps)

```bash
# 1. Clone repo
git clone https://github.com/AuroraHashcat/Causal-CoT.git

# 2. Enter repository
cd Causal-CoT

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the example
cd causal_cot
python stable_run.py --mode 1 --input-file ../causalnet/causalnet_llama-8b.csv
```

- `--mode 1` — example execution mode used for demonstration (adjust per project code comments).  
- `--input-file` — path to the dataset CSV (here we use the CausalNet example file).

Outputs (logs, graphs, DAG visualizations, and result summaries) are written to the configured `outputs/` or `results/` directory (see `stable_run.py` for exact paths).

---

## 📋 Usage / Arguments (example)

`stable_run.py` accepts (at least) these common arguments:

```text
--mode             Execution mode (int). Example: 1 = run full pipeline on dataset.
--input-file       Path to CSV/JSON dataset used by the experiment.
--model            LLM model identifier (optional; depends on your environment).
--output-dir       Directory for results/plots (optional).
--temperature      LLM temperature for prompting (optional).
--seed             Random seed for reproducibility (optional).
```

> See the script's top help for full, up-to-date options:
> `python stable_run.py --help`

---

## 🔬 Reproducibility & experiments

This repository aims to reproduce the main pipeline and representative results from the paper:

- Datasets evaluated: **MATH**, **CausalNet (CNET)**, **COPA**, **CSQA**, **GPQA**, **STRATEGYQA**, **HellaSwag**.  
- Typical evaluation: reformulate multiple-choice as binary causal judgments, construct/augment DAGs, estimate conditional probabilities via LLM prompting, apply do-calculus formulas (ATE / NDE / NIE / TE) and threshold (τ) to accept or reject causal links.  
- Example reported improvements (averaged over models in the paper):

| Domain       | Dataset   | CoT (%) | Causal-CoT (%) | Δ   |
|--------------|-----------|--------:|---------------:|----:|
| Math         | MATH      |   49.3  |          52.0  | +2.7|
| Causal       | CausalNet |   61.2  |          66.0  | +4.8|
| Commonsense  | GPQA      |   38.0  |          58.7  |+20.7|

> Exact reproduction requires the same LLMs, seeds, and (optional) retrieval backends; see `stable_run.py` and experiment config for details.

---

## 🛠 Implementation notes

- **DAG construction**: decompose premise/hypothesis into atomic statements and extract stated relations to form initial DAG (G₀).  
- **Augmentation**: run internal prompting (or IR retrieval + fusion) to add mediators, confounders, and colliders, producing (G_c).  
- **Verification**: use targeted prompts to obtain verbal likelihoods (e.g., “very unlikely” → calibrated probability), map them to numeric probabilities (via calibration table or Beta priors), then compute causal effects.  
- **Probabilities & calibration**: verbal-to-probability mapping uses calibrated buckets (e.g., very unlikely → 0.01–0.1, possible → 0.3–0.7).

---

## 🔍 Troubleshooting

- **Missing dependencies / import errors**: ensure `pip install -r requirements.txt`.  
- **LLM API access**: configure your environment variables or API keys as described in `stable_run.py`.  
- **Dataset path errors**: verify relative path for `--input-file` (e.g., `../causalnet/...`).  
- **Slow runs**: causal verification adds latency — use smaller datasets or debug mode for testing.

---

## 🧾 Citation

If you use this code or the ideas in your work, please cite:

```bibtex
@inproceedings{causalcot2026,
  title     = {Causal-CoT: Causal Chain-of-Thought for Validated Reasoning},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---

## 🔐 License

This repository is released under the **MIT License**. See `LICENSE` for details.

---

## 👥 Contributors

- **AuroraHashcat** — implementation, experiments, and integration.

---

## 📂 Acknowledgements & reproducibility statement

The approach builds upon Chain-of-Thought prompting and causal inference foundations. Supplementary datasets, scripts, and configurations are included in the repository under `docs/` and `causalnet/`.

---

## 📬 Contact / Issues

If you find bugs or wish to reproduce specific results, please open an issue or pull request at:  
**[https://github.com/AuroraHashcat/Causal-CoT](https://github.com/AuroraHashcat/Causal-CoT)**
