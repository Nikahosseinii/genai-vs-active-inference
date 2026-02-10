# ==============================================================
# Figure 5(b,c) — Relative characteristic path length & clustering
# AIF vs DDPM (snapshots every k rewiring steps)
#
# Computes for each snapshot graph:
#   L(s): characteristic path length (largest connected component)
#   C(s): average clustering coefficient
# Then normalizes:
#   L_rel(s) = L(s)/L(0)
#   C_rel(s) = C(s)/C(0)
#
# Example:
#   python scripts/path_length_and_clustering_aif_vs_ddpm.py ^
#     --aif_dir data/AIF/tensors --ddpm_dir data/DDPM/tensors --step_interval 100
# ==============================================================
import argparse
import os
import glob
import random
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
# Metrics: L, C
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

def char_path_length(g: nx.Graph, sample_k=30) -> float:
    """Characteristic path length on largest connected component."""
    if g.number_of_nodes() <= 1:
        return 0.0
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    return sampled_apsp(g, k=sample_k)

def clustering_coeff(g: nx.Graph) -> float:
    if g.number_of_nodes() <= 1:
        return 0.0
    return float(nx.average_clustering(g))

def compute_series(graphs, sample_k=30):
    L_vals = np.array([char_path_length(g, sample_k=sample_k) for g in graphs], dtype=float)
    C_vals = np.array([clustering_coeff(g) for g in graphs], dtype=float)

    # Normalize safely
    L0 = L_vals[0] if L_vals[0] != 0 else 1e-12
    C0 = C_vals[0] if C_vals[0] != 0 else 1e-12

    return (L_vals / L0), (C_vals / C0)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aif_dir", type=str, default="data/AIF/tensors", help="Folder containing AIF *.pt snapshots")
    ap.add_argument("--ddpm_dir", type=str, default="data/DDPM/tensors", help="Folder containing DDPM *.pt snapshots")
    ap.add_argument("--step_interval", type=int, default=100, help="Step distance between snapshots (x-axis)")
    ap.add_argument("--sample_k", type=int, default=30, help="Sample size for APSP estimate")
    ap.add_argument("--xmax", type=int, default=4000)
    ap.add_argument("--ylim_L", type=float, default=1.0, help="y-limit for relative path length panel")
    ap.add_argument("--ylim_C", type=float, default=1.6, help="y-limit for relative clustering panel")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--out_name", type=str, default="relative_path_length_and_clustering_aif_vs_ddpm.png")
    ap.add_argument("--seed", type=int, default=0)
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

    L_aif, C_aif = compute_series(aif_graphs, sample_k=args.sample_k)
    L_ddpm, C_ddpm = compute_series(ddpm_graphs, sample_k=args.sample_k)

    steps_aif = np.arange(len(L_aif)) * args.step_interval
    steps_ddpm = np.arange(len(L_ddpm)) * args.step_interval

    # Plot: two stacked panels, shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Panel 1: normalized path length
    ax1.plot(steps_aif, L_aif, lw=2, linestyle="-", label="AIF")
    ax1.plot(steps_ddpm, L_ddpm, lw=2, linestyle="--", label="DDPM")
    ax1.set_ylim(0, args.ylim_L)
    ax1.set_ylabel("Relative Characteristic Path Length")
    ax1.set_title("Relative Path Length vs. Network Rewiring Step")
    ax1.grid(ls=":", alpha=0.6)
    ax1.legend()

    # show x labels on top panel too (matches your figure style)
    ax1.set_xlabel("Network Rewiring Step")
    ax1.tick_params(axis="x", which="both", labelbottom=True)

    # Panel 2: normalized clustering
    ax2.plot(steps_aif, C_aif, lw=2, linestyle="-", label="AIF")
    ax2.plot(steps_ddpm, C_ddpm, lw=2, linestyle="--", label="DDPM")
    ax2.set_ylim(0, args.ylim_C)
    ax2.set_ylabel("Relative Clustering Coefficient")
    ax2.set_xlabel("Network Rewiring Step")
    ax2.set_title("Relative Clustering vs. Network Rewiring Step")
    ax2.grid(ls=":", alpha=0.6)
    ax2.legend()

    ax2.set_xlim(0, args.xmax)
    # Show ticks every 500 for readability (still supports step_interval=100)
    xticks = np.arange(0, args.xmax + args.step_interval, args.step_interval)
    ax2.set_xticks(xticks[::5])
    ax1.set_xticks(ax2.get_xticks())

    plt.tight_layout()
    out_path = out_dir / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
