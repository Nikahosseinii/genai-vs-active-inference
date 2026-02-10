# From Generative AI to Active Inference: Enhancing Human–AI Interaction via Model Reduction, Refinement, and Unification

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
└── README.md'''
Dataset
This project uses the Scale-Free Small-World Networks (64 nodes) dataset.

Download
Please download the dataset from IEEE DataPort:

https://ieee-dataport.org/documents/scale-free-small-world-networks-64-nodes

The dataset consists of adjacency matrices representing scale-free networks with small-world properties.

Generating Diffusion Trajectories and Samples

All diffusion experiments start from the same seed graph, provided as:

data/x0.pt


This file contains the initial ring-lattice / small-world seed adjacency matrix used to initialize the DDPM forward diffusion process.

Forward Diffusion (Trajectory Generation)

To generate diffusion trajectories:

Load data/x0.pt as the initial adjacency tensor.

Run the DDPM forward diffusion process starting from this seed graph.

Save intermediate adjacency tensors every 100 diffusion steps (e.g., steps 0, 100, 200, …, 4000).

These saved tensors correspond to noisy graph states 
𝑥
𝑡
x
t
	​

 along the forward diffusion trajectory and can be used for structural analysis or visualization.

Reverse Diffusion (Sampling)

For reverse diffusion (graph generation), users have two options:

Use the provided pretrained checkpoint
Download the checkpoint linked above and sample graphs by running the reverse denoising process starting from pure noise, as in standard DDPM sampling.

Train the DDPM model from scratch
Alternatively, users may retrain the DDPM using the provided dataset and scripts, then sample graphs from their own trained checkpoint.

In both cases, generated adjacency tensors can be saved every 100 reverse steps to obtain the full denoising trajectory.

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
Our DDPM baseline is adapted from OpenAI’s Improved DDPM implementation:
https://github.com/openai/improved-diffusion

We modified the original image-based DDPM to operate on graph adjacency matrices instead of RGB images. In particular, we replace image tensors with symmetric binary adjacency tensors, apply thresholding to convert continuous outputs into discrete graph edges, and analyze each generated tensor as a NetworkX graph. These adaptations allow DDPM to generate scale-free, small-world network structures, which are then used as a non-agentic generative baseline for comparison with Active Inference–based network rewiring.


The original implementation is designed for image generation using RGB pixel grids. In this work, we adapt the model for graph generation and network rewiring, inspired by the generative dynamics of mycorrhizal (fungal) networks.

Conceptually, we interpret DDPM’s forward diffusion process as biomimicking the branching and exploratory growth of fungi, while the reverse denoising process biomimics fusion and structural consolidation. Graphs sampled from the learned reverse process exhibit power-law degree distributions and low average path lengths, reflecting the well-known combination of scale-free hubs and small-world organization observed in natural mycorrhizal networks.

No VS Code configuration is required

No launch.json is required

All scripts run directly from the command line

Installation
Create a Python environment and install the required dependencies:

'''pip install -r DDPM/requirements.txt
Required Packages
torch

numpy

networkx

mpi4py

blobfile'''

Training DDPM
Train a DDPM model on the graph dataset.

macOS / Linux
'''python DDPM/image_train.py \
  --data_dir data/scale_free_small_world_64 \
  --image_size 64 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --diffusion_steps 4000 \
  --noise_schedule linear \
  --lr 1e-4'''
Windows (PowerShell)
'''python DDPM\image_train.py `
  --data_dir data\scale_free_small_world_64 `
  --image_size 64 `
  --num_channels 128 `
  --num_res_blocks 3 `
  --diffusion_steps 4000 `
  --noise_schedule linear `
  --lr 1e-4'''
Sampling Graphs with DDPM
After training, generate new graphs using a trained checkpoint.

macOS / Linux
'''python DDPM/image_sample.py \
  --model_path checkpoints/ema_0.9999_XXXXXX.pt \
  --image_size 64 \
  --num_channels 128 \
  --num_res_blocks 3 \
  --diffusion_steps 4000 \
  --noise_schedule linear'''
Windows (PowerShell)
'''python DDPM\image_sample.py `
  --model_path checkpoints\ema_0.9999_XXXXXX.pt `
  --image_size 64 `
  --num_channels 128 `
  --num_res_blocks 3 `
  --diffusion_steps 4000 `
  --noise_schedule linear'''
Replace XXXXXX with the checkpoint step number.

Pretrained DDPM Checkpoint

We provide a pretrained DDPM checkpoint for graph generation on 64-node scale-free small-world networks.

Model: Improved DDPM adapted for graph adjacency matrices

Diffusion steps: 4000

Training data: IEEE DataPort scale-free small-world networks

Purpose: Non-agentic generative baseline for comparison with Active Inference

Download: https://drive.google.com/file/d/1YIePTwQaTfC__svLVfMpoySjJXUnG5gu/view?usp=sharing

After downloading, place the checkpoint in the repository as:
'''
checkpoints/
└── ema_0.9999_1610000.pt'''

The pretrained checkpoint can be downloaded from Google Drive:

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