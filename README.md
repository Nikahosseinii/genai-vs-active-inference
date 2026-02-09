\# Generative AI vs Active Inference (Social Contagion Benchmark)



Reference implementation for the paper:

\*\*From Generative AI to Active Inference: Enhancing Human–AI Interaction via Model Reduction, Refinement, and Unification\*\*



This repository provides code to run an Active Inference–inspired network rewiring benchmark and measure:

\- social contagion time (time to full adoption)

\- teleological token actuation (token “bill”)

under different spreading probabilities r.



\## Repository structure

\- `src/utils/` : utilities (e.g., initial graph generation)

\- `scripts/`   : runnable experiments/benchmarks

\- `data/`      : generated inputs (e.g., `x0.pt`)

\- `figures/`   : saved plots



\## Requirements

Install dependencies:

```bash

pip install -r requirements.txt



