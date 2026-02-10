# Generative AI vs Active Inference (Social Contagion Benchmark)

Reference implementation for the paper:

**From Generative AI to Active Inference: Enhancing Human–AI Interaction via Model Reduction, Refinement, and Unification**

This repository provides a benchmark framework for comparing **Active Inference (AIF)** and **Generative AI (DDPM-based)** approaches to network reconfiguration and social contagion dynamics.

The code enables simulation and evaluation of:
- **Social contagion time** (time to full adoption)
- **Teleological token actuation** (token “bill” accumulation)
- Structural network evolution under varying spreading probabilities \( r \)

---

## Repository Structure

```text
.
├── src/
│   └── utils/            # Utility functions (e.g., graph initialization)
├── scripts/              # Runnable experiments and benchmarks
├── DDPM/                 # DDPM baseline (graph generation)
├── data/                 # Input datasets (e.g., x0.pt)
├── figures/              # Generated plots and figures
└── README.md
Dataset
This project uses the Scale-Free Small-World Networks (64 nodes) dataset.

Download
Please download the dataset from IEEE DataPort:

https://ieee-dataport.org/documents/scale-free-small-world-networks-64-nodes

The dataset consists of adjacency matrices representing scale-free networks with small-world properties.

Expected Directory Structure
After downloading and extracting the dataset, place it in the repository as follows:

data/
└── scale_free_small_world_64/
    ├── x0.pt
    ├── sample_001.pt
    ├── sample_002.pt
    └── ...
Notes

Each .pt file must contain a 64 × 64 adjacency matrix

Non-zero entries are treated as edges

If your filenames differ, update the corresponding paths in the scripts

DDPM Baseline (Graph Generation)
We include a DDPM baseline for graph generation, adapted from standard diffusion models and used only as a comparison method.

No VS Code configuration is required

No launch.json is required

All scripts run directly from the command line

Installation
Create a Python environment and install the required dependencies:

pip install -r DDPM/requirements.txt
Required Packages
torch

numpy

networkx

mpi4py

blobfile

Training DDPM
Train a DDPM model on the graph dataset.

macOS / Linux
python DDPM/image_train.py \
  --data_dir data/scale_free_small_world_64 \
  --image_size 64 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --diffusion_steps 4000 \
  --noise_schedule linear \
  --lr 1e-4
Windows (PowerShell)
python DDPM\image_train.py `
  --data_dir data\scale_free_small_world_64 `
  --image_size 64 `
  --num_channels 128 `
  --num_res_blocks 3 `
  --diffusion_steps 4000 `
  --noise_schedule linear `
  --lr 1e-4
Sampling Graphs with DDPM
After training, generate new graphs using a trained checkpoint.

macOS / Linux
python DDPM/image_sample.py \
  --model_path checkpoints/ema_0.9999_XXXXXX.pt \
  --image_size 64 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --diffusion_steps 4000 \
  --noise_schedule linear
Windows (PowerShell)
python DDPM\image_sample.py `
  --model_path checkpoints\ema_0.9999_XXXXXX.pt `
  --image_size 64 `
  --num_channels 128 `
  --num_res_blocks 3 `
  --diffusion_steps 4000 `
  --noise_schedule linear
Replace XXXXXX with the checkpoint step number.

Active Inference Benchmark
The Active Inference implementation includes:

Wake-only dynamics

Wake–sleep and wake–rest cycles

Bayesian Model Reduction (BMR)

Entropy-maximizing rest

Precision modulation (including psychedelic intervention experiments)

These scripts reproduce the structural and dynamical results reported in the paper, including:

Small-world index

Characteristic path length

Clustering coefficient

Prior entropy

Posterior synchrony

Confidence dynamics