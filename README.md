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



Dataset

This project uses the Scale-Free Small-World Networks (64 nodes) dataset.

Download

Please download the dataset from IEEE DataPort:

👉 https://ieee-dataport.org/documents/scale-free-small-world-networks-64-nodes

The dataset contains adjacency matrices representing scale-free networks with small-world properties.

Expected directory structure

After downloading and extracting the dataset, place it in the repository as follows:

data/
└── scale_free_small_world_64/
    ├── x0.pt
    ├── *.pt


Each .pt file should contain a 64×64 adjacency matrix

Non-zero entries are treated as edges

If your file names differ, simply update the paths in the scripts.

DDPM Baseline (Graph Generation)

We include a DDPM baseline for graph generation, adapted from standard diffusion models and used here only as a comparison method.

No VS Code or launch.json is required.
All scripts run directly from the command line.

Installation

Create a Python environment and install dependencies:

pip install -r DDPM/requirements.txt


Required packages include:

torch

numpy

networkx

mpi4py

blobfile

Training DDPM

Train a DDPM model on the graph dataset:

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

After training, generate new graphs using the trained checkpoint:

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