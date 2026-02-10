# ==============================================================
# Figure 10 — Wake vs Rest accuracy curves
#
# Wake  ≔ free-energy minimisation (greedy edge additions)
# Rest  ≔ spatio-temporal-entropy maximisation (accept only if ↑H_S and ↑H_T)
#
# Plots "Accuracy" proxy as  -F  (negative variational free energy).
#
# Usage (Watts–Strogatz seed):
#   python scripts/aif_wake_rest_accuracy.py
#
# Usage (use your x0.pt adjacency tensor):
#   python scripts/aif_wake_rest_accuracy.py --use_x0 --pt_path data/x0.pt
# ==============================================================
import argparse
import random
from pathlib import Path
from math import log2

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Repo root + output dir
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ───────── 1) Load seed graph ──────────────────────────────────
def load_x0_graph(pt_path: Path) -> nx.Graph:
    import torch
    torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
    adj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    elif isinstance(adj, dict) and "adj" in adj:
        adj = adj["adj"]
    adj = (np.array(adj) > 0).astype(int)
    return nx.from_numpy_array(adj)

def set_bipartition_clusters(G: nx.Graph, attr="cluster"):
    """Simple deterministic bipartition (first half vs second half)."""
    n = G.number_of_nodes()
    for v in G.nodes:
        G.nodes[v][attr] = 0 if v < n // 2 else 1

# ───────── 2) Generative-model scaffolding ─────────────────────
def make_full_params(N):
    pointy = np.full(N, 0.4 / (N - 1))
    pointy[0] = 0.6
    D = dict(norm=np.array([0.4, 0.6]),
             MB=np.array([0.4, 0.6]),
             net=pointy,
             tok=np.array([0.4, 0.6]))
    off = 0.1 / (N - 1)
    net_like = np.full((N, N), off)
    np.fill_diagonal(net_like, 0.9)
    A = dict(norm=np.array([[0.9, 0.1], [0.1, 0.9]]),
             MB=np.array([[0.8, 0.2], [0.2, 0.8]]),
             net=net_like,
             tok=np.array([[0.85, 0.15], [0.15, 0.85]]))
    return D, A

def posterior_Q(g, D):
    Q = {k: d / d.sum() for k, d in D.items()}
    deg = np.asarray([d for _, d in g.degree()], float)
    Q["net"] = deg / deg.sum() if deg.sum() else np.ones_like(deg) / len(deg)
    return Q

def free_energy(Q, D, A, obs, eps=1e-32):
    """Complexity − accuracy."""
    F = 0.0
    for f in ("norm", "MB", "net", "tok"):
        q, d = Q[f], D[f]
        F += np.sum(q * (np.log(q + eps) - np.log(d + eps)))     # KL term
        F -= np.sum(q * np.log(A[f][obs[f]] + eps))              # accuracy
    return float(F)

# ───────── 3) Entropy helpers ──────────────────────────────────
def spatial_entropy(g, attr="cluster"):
    """Entropy of cluster sizes (demo proxy)."""
    clusters, N = {}, g.number_of_nodes()
    for v, d in g.nodes(data=True):
        clusters.setdefault(d[attr], []).append(v)
    return -sum((len(C) / N) * log2(len(C) / N) for C in clusters.values() if len(C))

def temporal_entropy(history, clusters):
    """Entropy of cluster membership among currently 'active' nodes."""
    active = {v for v, h in history.items() if h}
    if len(active) < 2:
        return 0.0
    counts = {}
    for v in active:
        counts[clusters[v]] = counts.get(clusters[v], 0) + 1
    N = len(active)
    return -sum((c / N) * log2(c / N) for c in counts.values())

# ───────── 4) Scheduler — wake minimises F; rest maximises H_S & H_T ─────────
def run_schedule(
    G0,
    *,
    steps=2000,
    rest_every=3,
    rest_replays=5,
    add_edges=2,
    p_wake=0.02,
    p_rest=0.20,
    seed=1,
    cluster_attr="cluster",
):
    rng = random.Random(seed)
    g = G0.copy()
    D, A = make_full_params(g.number_of_nodes())

    clusters = {v: d[cluster_attr] for v, d in g.nodes(data=True)}
    history = {v: [] for v in g.nodes()}
    obs_stub = dict(norm=0, MB=0, net=0, tok=0)

    H_S = spatial_entropy(g, attr=cluster_attr)
    H_T = temporal_entropy(history, clusters)

    acc = []

    for t in range(steps):
        phase = "wake" if (t % rest_every) < rest_every - 1 else "rest"
        p_now = p_wake if phase == "wake" else p_rest

        # ---- WAKE: minimise free energy (greedy) ------------------------
        if phase == "wake":
            non_edges = list(nx.non_edges(g))
            for _ in range(add_edges):
                if not non_edges:
                    break
                cand = rng.sample(non_edges, min(3, len(non_edges)))
                F0 = free_energy(posterior_Q(g, D), D, A, obs_stub)
                best, best_delta = None, 0.0

                for u, v in cand:
                    g.add_edge(u, v)
                    delta = free_energy(posterior_Q(g, D), D, A, obs_stub) - F0
                    if delta < best_delta:  # largest decrease in F
                        best, best_delta = (u, v), delta
                    g.remove_edge(u, v)

                if best:
                    g.add_edge(*best)
                    if best in non_edges:
                        non_edges.remove(best)

        # ---- REST: accept a move only if ↑H_S AND ↑H_T ------------------
        else:
            for _ in range(rest_replays):
                best_move = None
                base_HS, base_HT = H_S, H_T

                for act in ("add", "remove", "swap"):
                    tmp = g.copy()

                    if act == "add":
                        ne = list(nx.non_edges(tmp))
                        if ne:
                            tmp.add_edge(*rng.choice(ne))
                    elif act == "remove" and tmp.number_of_edges():
                        tmp.remove_edge(*rng.choice(list(tmp.edges)))
                    elif act == "swap" and tmp.number_of_edges():
                        tmp.remove_edge(*rng.choice(list(tmp.edges)))
                        ne = list(nx.non_edges(tmp))
                        if ne:
                            tmp.add_edge(*rng.choice(ne))

                    HS_new = spatial_entropy(tmp, attr=cluster_attr)
                    if HS_new <= base_HS:
                        continue

                    hist_tmp = {v: h.copy() for v, h in history.items()}
                    for v in tmp.nodes:
                        if rng.random() < p_rest:
                            hist_tmp[v].append(t)
                    HT_new = temporal_entropy(hist_tmp, clusters)
                    if HT_new <= base_HT:
                        continue

                    best_move = (tmp, HS_new, HT_new, hist_tmp)
                    break  # first move that passes both tests

                if best_move:
                    g, H_S, H_T, history = best_move

        # ---- norm-switch events (all phases) ---------------------------
        for v in g.nodes:
            if rng.random() < p_now:
                history[v].append(t)
        H_T = temporal_entropy(history, clusters)

        # "Accuracy" proxy: E_Q[log p] ~ -F
        acc.append(-free_energy(posterior_Q(g, D), D, A, obs_stub))

    return np.asarray(acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--rest_every", type=int, default=3)
    ap.add_argument("--rest_replays", type=int, default=5)
    ap.add_argument("--add_edges", type=int, default=2)

    ap.add_argument("--p_wake", type=float, default=0.02)
    ap.add_argument("--p_rest", type=float, default=0.20)

    ap.add_argument("--use_x0", action="store_true")
    ap.add_argument("--pt_path", type=str, default="data/x0.pt")
    ap.add_argument("--out_name", type=str, default="accuracy_wake_vs_rest.png")
    args = ap.parse_args()

    # Seed graph
    if args.use_x0:
        pt = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
        G_seed = load_x0_graph(pt)
    else:
        N_nodes = 32
        G_seed = nx.watts_strogatz_graph(N_nodes, k=4, p=0.1, seed=42)

    set_bipartition_clusters(G_seed, attr="cluster")

    # Run: wake-only vs wake+rest
    acc_wake = run_schedule(G_seed, steps=args.steps, rest_every=999999, seed=args.seed,
                            rest_replays=args.rest_replays, add_edges=args.add_edges,
                            p_wake=args.p_wake, p_rest=args.p_rest)

    acc_rest = run_schedule(G_seed, steps=args.steps, rest_every=args.rest_every, seed=args.seed,
                            rest_replays=args.rest_replays, add_edges=args.add_edges,
                            p_wake=args.p_wake, p_rest=args.p_rest)

    # Plot + save
    plt.figure(figsize=(8, 4))
    plt.plot(acc_wake, "r--", label="Wake only")
    plt.plot(acc_rest, "b", label="Wake + Rest")
    plt.xlabel("Network rewiring steps", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlim(0, args.steps)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
