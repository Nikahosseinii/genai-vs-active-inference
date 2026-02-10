# ==============================================================
# Active Inference — Wake-only vs Wake + Rest (entropy-maximizing)
#
# Logs spatial entropy H_S over network rewiring steps and saves a
# figure suitable for Figure 9(a)-style results in the paper.
#
# Usage (Watts–Strogatz seed, default):
#   python scripts/aif_wake_rest_spatial_entropy.py
#
# Usage (use your x0.pt adjacency tensor):
#   python scripts/aif_wake_rest_spatial_entropy.py --use_x0 --pt_path data/x0.pt
# ==============================================================
import argparse
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy

# Repo root + output dir
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Helpers: free energy proxy, posterior, spatial entropy
# ────────────────────────────────────────────────────────────────
def free_energy(Q, D, A, obs, eps=1e-32):
    """Complexity - accuracy (proxy used in these benchmark scripts)."""
    F = 0.0
    for f in ["norm", "MB", "net", "tok"]:
        q, d = Q[f], D[f]
        F += np.sum(q * (np.log(q + eps) - np.log(d + eps)))   # complexity
        F -= np.sum(q * np.log(A[f][obs[f]] + eps))            # accuracy
    return float(F)

def posterior_Q(g, D):
    """Approximate posterior; 'net' factor proxy via normalized degrees."""
    Q = {k: d / d.sum() for k, d in D.items()}
    deg = np.array([d for _, d in g.degree()], float)
    Q["net"] = deg / deg.sum() if deg.sum() else np.ones_like(deg) / len(deg)
    return Q

def spatial_entropy(g: nx.Graph) -> float:
    """Shannon entropy of shortest-path-length distribution."""
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()

    dists = []
    sp = dict(nx.shortest_path_length(g))
    for u in sp:
        for v, d in sp[u].items():
            if u < v:
                dists.append(d)

    counts = np.bincount(dists)
    probs = counts / counts.sum()
    return float(shannon_entropy(probs))

# ────────────────────────────────────────────────────────────────
# Priors & likelihoods (kept consistent with your other scripts)
# ────────────────────────────────────────────────────────────────
def make_full_params(N):
    pointy = np.full(N, 0.4 / (N - 1))
    pointy[0] = 0.6
    D = dict(
        norm=np.array([0.4, 0.6]),
        MB=np.array([0.4, 0.6]),
        net=pointy,
        tok=np.array([0.4, 0.6]),
    )

    off = 0.1 / (N - 1)
    net_like = np.full((N, N), off)
    np.fill_diagonal(net_like, 0.9)
    A = dict(
        norm=np.array([[0.9, 0.1], [0.1, 0.9]]),
        MB=np.array([[0.8, 0.2], [0.2, 0.8]]),
        net=net_like,
        tok=np.array([[0.85, 0.15], [0.15, 0.85]]),
    )
    return D, A

# ────────────────────────────────────────────────────────────────
# ACTIVE-INFERENCE SCHEDULER: wake vs rest
# ────────────────────────────────────────────────────────────────
def run_schedule(
    G0,
    *,
    steps=2000,
    rest_every=3,
    rest_replays=5,
    add_edges=2,
    seed=1,
):
    """
    Wake phase: FE-driven edge additions (sample 3 candidates; pick best ΔF).
    Rest phase: entropy-maximizing replays (random rewiring) + Dirichlet-like updates.
    """
    rng = random.Random(seed)
    g = G0.copy()
    D, A = make_full_params(g.number_of_nodes())
    obs_stub = dict(norm=0, MB=0, tok=0, net=0)

    H_S = []

    for t in range(steps):
        phase = "wake" if (t % rest_every) < rest_every - 1 else "rest"

        # ---------------- Wake ----------------
        if phase == "wake":
            non_edges = list(nx.non_edges(g))
            for _ in range(add_edges):
                if not non_edges:
                    break
                cand = rng.sample(non_edges, min(3, len(non_edges)))
                Q0 = posterior_Q(g, D)
                F0 = free_energy(Q0, D, A, obs_stub)
                best, best_delta = None, 0.0
                for u, v in cand:
                    g.add_edge(u, v)
                    delta = free_energy(posterior_Q(g, D), D, A, obs_stub) - F0
                    if delta < best_delta:
                        best, best_delta = (u, v), delta
                    g.remove_edge(u, v)
                if best:
                    g.add_edge(*best)
                    if best in non_edges:
                        non_edges.remove(best)

        # ---------------- Rest ----------------
        else:
            for _ in range(rest_replays):
                tmp = g.copy()

                action = rng.choice(["add", "remove", "swap"])
                if action == "add":
                    ne = list(nx.non_edges(tmp))
                    if ne:
                        tmp.add_edge(*rng.choice(ne))
                elif action == "remove" and tmp.number_of_edges():
                    tmp.remove_edge(*rng.choice(list(tmp.edges)))
                elif action == "swap" and tmp.number_of_edges():
                    tmp.remove_edge(*rng.choice(list(tmp.edges)))
                    ne = list(nx.non_edges(tmp))
                    if ne:
                        tmp.add_edge(*rng.choice(ne))

                # keep latest replay (simple entropy-max replay proxy)
                g = tmp

        H_S.append(spatial_entropy(g))

    return np.array(H_S)

def load_x0_graph(pt_path: Path) -> nx.Graph:
    """Load adjacency tensor from .pt and return as NetworkX graph."""
    import torch
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    adj = obj.numpy() if hasattr(obj, "numpy") else np.array(obj)
    return nx.from_numpy_array(adj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rest_every", type=int, default=3)
    ap.add_argument("--rest_replays", type=int, default=5)
    ap.add_argument("--add_edges", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--use_x0", action="store_true", help="Use data/x0.pt instead of Watts–Strogatz seed")
    ap.add_argument("--pt_path", type=str, default="data/x0.pt", help="Path to x0 adjacency tensor")
    ap.add_argument("--out_name", type=str, default="spatial_entropy_wake_vs_rest.png")

    args = ap.parse_args()

    # Choose seed graph
    if args.use_x0:
        pt = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
        G_seed = load_x0_graph(pt)
    else:
        N_nodes = 32
        G_seed = nx.watts_strogatz_graph(N_nodes, k=4, p=0.1, seed=42)

    # Run: wake-only (never rests) vs wake+rest
    wake_only = run_schedule(
        G_seed,
        steps=args.steps,
        rest_every=999999,     # effectively never rests
        rest_replays=args.rest_replays,
        add_edges=args.add_edges,
        seed=args.seed,
    )
    wake_rest = run_schedule(
        G_seed,
        steps=args.steps,
        rest_every=args.rest_every,
        rest_replays=args.rest_replays,
        add_edges=args.add_edges,
        seed=args.seed,
    )

    # Plot + save
    plt.figure(figsize=(9, 4))
    plt.plot(wake_only, "r--", label="Wake only")
    plt.plot(wake_rest, "b", label="Wake + Rest")
    plt.xlim(0, args.steps)
    plt.xlabel("Network rewiring steps")
    plt.ylabel("Spatial entropy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
