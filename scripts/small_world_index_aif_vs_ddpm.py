# ==============================================================
# Small-world index σ(s) — AIF vs DDPM (snapshots every k steps)
#
# Computes:
#   L(s): average shortest path length (largest connected component)
#   C(s): average clustering coefficient
#   σ(s) = (C(s)/C(0)) / (L(s)/L(0))
#
# Expects folders with *.pt adjacency tensors for each method.
#
# Example:
#   python scripts/small_world_index_aif_vs_ddpm.py ^
#     --aif_dir data/AIF/tensors --ddpm_dir data/DDPM/tensors --step_interval 100
# ==============================================================
import argparse
import os
import glob
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt


# ------------------------------
# Loading utilities
# ------------------------------
def load_adj_pt(path: Path) -> nx.Graph:
    torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
    obj = torch.load(path, map_location="cpu", weights_only=False)
    arr = obj.numpy() if isinstance(obj, torch.Tensor) else np.array(obj)
    return nx.from_numpy_array(arr)

def load_series(folder: Path):
    files = sorted(glob.glob(os.path.join(str(folder), "*.pt")))
    if not files:
        raise FileNotFoundError(f"No *.pt files found in: {folder}")
    return [load_adj_pt(Path(p)) for p in files]

# ------------------------------
# Metrics: L, C, sigma
# ------------------------------
def sampled_apsp(g: nx.Graph, k=30) -> float:
    """Sampled approximation to average shortest path length."""
    n = g.number_of_nodes()
    if n <= 1:
        return 0.0
    if k >= n:
        return float(nx.average_shortest_path_length(g))

    tot = 0.0
    cnt = 0
    nodes = list(g.nodes())
    for v in random.sample(nodes, k):
        d = nx.single_source_shortest_path_length(g, v)
        tot += float(sum(d.values()))
        cnt += (len(d) - 1)
    return tot / cnt if cnt > 0 else 0.0

def L(g: nx.Graph, sample_k=30) -> float:
    """Average shortest path length on largest connected component."""
    if g.number_of_nodes() <= 1:
        return 0.0
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    # NetworkX APSP is exact and can be slow; use sampled estimate by default.
    return sampled_apsp(g, k=sample_k)

def C(g: nx.Graph) -> float:
    """Average clustering coefficient."""
    if g.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_clustering(g))

def sigma_series(graphs, sample_k=30):
    Ls = np.array([L(g, sample_k=sample_k) for g in graphs], dtype=float)
    Cs = np.array([C(g) for g in graphs], dtype=float)

    # Avoid division by zero
    L0 = Ls[0] if Ls[0] != 0 else 1e-12
    C0 = Cs[0] if Cs[0] != 0 else 1e-12
    return (Cs / C0) / (Ls / L0)

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aif_dir", type=str, default="data/AIF/tensors", help="Folder containing AIF *.pt snapshots")
    ap.add_argument("--ddpm_dir", type=str, default="data/DDPM/tensors", help="Folder containing DDPM *.pt snapshots")
    ap.add_argument("--step_interval", type=int, default=100, help="Step distance between snapshots (x-axis)")
    ap.add_argument("--sample_k", type=int, default=30, help="Number of source nodes to sample for APSP estimate")
    ap.add_argument("--xmax", type=int, default=4000)
    ap.add_argument("--ymax", type=float, default=12.0)
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--out_name", type=str, default="small_world_index_aif_vs_ddpm.png")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for APSP sampling")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ROOT = Path(__file__).resolve().parents[1]
    aif_dir = (ROOT / args.aif_dir) if not Path(args.aif_dir).is_absolute() else Path(args.aif_dir)
    ddpm_dir = (ROOT / args.ddpm_dir) if not Path(args.ddpm_dir).is_absolute() else Path(args.ddpm_dir)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    aif_graphs = load_series(aif_dir)
    ddpm_graphs = load_series(ddpm_dir)

    sigma_aif = sigma_series(aif_graphs, sample_k=args.sample_k)
    sigma_ddpm = sigma_series(ddpm_graphs, sample_k=args.sample_k)

    steps_aif = np.arange(len(aif_graphs)) * args.step_interval
    steps_ddpm = np.arange(len(ddpm_graphs)) * args.step_interval

    for label, s in [("AIF", steps_aif), ("DDPM", steps_ddpm)]:
        if len(s) and s[-1] < args.xmax:
            warnings.warn(f"{label} stops at step {s[-1]} (<{args.xmax}).")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps_aif, sigma_aif, lw=2, linestyle="-", label="AIF")
    ax.plot(steps_ddpm, sigma_ddpm, lw=2, linestyle="--", label="DDPM")

    ax.set_xlim(0, args.xmax)
    ax.set_ylim(0, args.ymax)
    ax.set_xlabel("Network rewiring step", fontsize=12)
    ax.set_ylabel("Small-world index", fontsize=12)
    ax.set_title("AIF vs. DDPM", fontsize=14)
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()

    out_path = out_dir / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
