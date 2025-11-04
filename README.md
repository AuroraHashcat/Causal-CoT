
## Causal-CoT: Causal Chain-of-Thought for Validated Reasoning

Official implementation of **Causal-CoT**, a framework that integrates causal graph construction, augmentation, and verification into the Chain-of-Thought (CoT) paradigm.  
Paper: *Causal-CoT: Causal Chain-of-Thought for Validated Reasoning* (under review at ICLR 2026)

### Overview

Standard CoT prompting enables large language models (LLMs) to produce step-by-step reasoning, yet the intermediate rationales are often unfaithful or logically inconsistent. Causal‑CoT addresses this by introducing causal graph–based reasoning:

- **DAG-guided CoT** — construct an initial directed acyclic graph (DAG) from the problem context.  
- **Reflection & Augmentation** — enrich the DAG by adding plausible mediators or contextual variables.  
- **Causal Verification** — estimate conditional probabilities via prompting and apply do-calculus to validate causal effects.

<p align="center">
  <img src="docs/causal_cot_pipeline.png" width="600" alt="Causal-CoT Pipeline"/>
</p>

This structured reasoning converts linear CoT chains into verifiable causal graphs, improving both reasoning fidelity and interpretability across mathematics, commonsense, and causal inference tasks.

### Key Features

- **Graph-Structured Reasoning:** Each reasoning trace forms a DAG.  
- **Integrated do-calculus Verification:** Quantitative validation of causal links.  
- **Modular Pipeline:** Three-stage workflow (DAG → Augmentation → Verification).  
- **Benchmark Coverage:** Evaluated on 7 benchmarks (MATH, CausalNet, COPA, CSQA, GPQA, STRATEGYQA, HellaSwag).  
- **Cross-Model Compatibility:** Works with LLaMA-3, Qwen-2.5, GPT-3.5, Claude-3.5/3.7, DeepSeek-R1, O3-Mini, etc.

### Installation

```bash
git clone https://github.com/AuroraHashcat/Causal-CoT.git
cd Causal-CoT

# Create environment and install dependencies
pip install -r requirements.txt
```

### Usage Example

```bash
# Move into main module directory
cd causal_cot

# Run a causal reasoning experiment
python stable_run.py --mode 1 --input-file ../causalnet/causalnet_llama-8b.csv
```

Arguments:

- `--mode`: execution mode (e.g., `1` for standard run)  
- `--input-file`: path to the dataset CSV (e.g., `causalnet_llama-8b.csv`)

Results (plots / logs / graphs) are saved automatically under the output directory.

### Project Structure (excerpt)

```
Causal-CoT/
├── causal_cot/                # Core pipeline implementation
│   ├── stable_run.py
│   ├── dag_construction.py
│   ├── augmentation.py
│   └── verification.py
├── causalnet/                 # Benchmark datasets
├── docs/                      # Documentation & figures
├── requirements.txt
└── README.md
```

### Reproduction (selected results)

| Domain     | Dataset     | Baseline CoT (%) | Causal-CoT (%) | Δ (↑)  |
|------------|-------------:|------------------:|---------------:|-------:|
| Math       | MATH         | 49.3              | 52.0           | +2.7   |
| Causal     | CausalNet    | 61.2              | 66.0           | +4.8   |
| Commonsense| GPQA         | 38.0              | 58.7           | +20.7  |

Causal-CoT consistently improves reasoning fidelity and stability across multiple LLMs while maintaining a favorable accuracy–efficiency trade-off.

### Citation

If you use this work, please cite:

```
@inproceedings{causalcot2026,
  title     = {Causal-CoT: Causal Chain-of-Thought for Validated Reasoning},
  author    = {Anonymous},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

### Contributors & License

- Implementation and experiments: AuroraHashcat  
- License: MIT — see LICENSE for details
