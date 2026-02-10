# ==============================================================
# Active Inference — Wake-only vs Wake + Rest (entropy-maximizing)
#
# Logs:
#   - Spatial entropy H_S  (path-length entropy)
#   - Temporal entropy H_T (inter-event gap entropy)
#   - Spatio-temporal entropy H_ST (joint entropy of path-length × gap)
#
# Intended for Figure 9(b)-style results in the paper.
#
# Usage (default: Watts–Strogatz seed):
#   python scripts/aif_wake_rest_spatiotemporal_entropy.py
#
# Usage (use your x0.pt adjacency tensor):
#   python scripts/aif_wake_rest_spatiotemporal_entropy.py --use_x0 --pt_path data/x0.pt
# ==============================================================
import argparse
import random
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy

# Repo root + output dir
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Entropy helpers
# ────────────────────────────────────────────────────────────────
def spatial_entropy(g: nx.Graph) -> float:
    """Path-length entropy (all-pairs shortest paths, natural log)."""
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

def temporal_entropy(event_times, window=20) -> float:
    """Entropy of inter-event gaps over a sliding window."""
    if len(event_times) < 2:
        return 0.0
    gaps = np.diff(event_times[-window:])
    if gaps.size == 0:
        return 0.0
    _, cnts = np.unique(gaps, return_counts=True)
    return float(shannon_entropy(cnts / cnts.sum()))

def spatio_temporal_entropy(g: nx.Graph, event_times, window=20) -> float:
    """Joint entropy of shortest-path lengths × inter-event gaps."""
    if len(event_times) < 2:
        return 0.0

    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()

    # spatial distribution of path lengths
    dists = []
    sp = dict(nx.shortest_path_length(g))
    for u in sp:
        for v, d in sp[u].items():
            if u < v:
                dists.append(d)
    cd = np.bincount(dists)
    pd = cd / cd.sum()

    # temporal distribution of gaps (last window)
    gaps = np.diff(event_times[-window:])
    if gaps.size == 0:
        return 0.0
    _, ct = np.unique(gaps, return_counts=True)
    pt = ct / ct.sum()

    # independent joint via outer product
    joint = np.outer(pd, pt)
    return float(shannon_entropy(joint.ravel()))

# ────────────────────────────────────────────────────────────────
# Minimal FE proxy machinery (kept consistent with other scripts)
# ────────────────────────────────────────────────────────────────
def posterior_Q(g, D):
    Q = {k: d / d.sum() for k, d in D.items()}
    deg = np.asarray([d for _, d in g.degree()], float)
    Q["net"] = deg / deg.sum() if deg.sum() else np.ones_like(deg) / len(deg)
    return Q

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

def free_energy(Q, D, A, obs, eps=1e-32):
    F = 0.0
    for f in ["norm", "MB", "net", "tok"]:
        q, d = Q[f], D[f]
        F += np.sum(q * (np.log(q + eps) - np.log(d + eps)))  # complexity
        F -= np.sum(q * np.log(A[f][obs[f]] + eps))           # accuracy
    return float(F)

# ────────────────────────────────────────────────────────────────
# ACTIVE-INFERENCE SCHEDULER (logs H_S, H_T, H_ST)
# ────────────────────────────────────────────────────────────────
def run_schedule(
    G0,
    *,
    steps=2000,
    start_rest=500,
    rest_every=3,
    rest_replays=5,
    add_edges=2,
    p_event=0.3,
    seed=1,
    gap_window=20,
):
    """
    Wake phase: FE-driven edge additions (sample 3 candidates; pick best ΔF).
    Rest phase: entropy-maximizing replays (random rewiring).
    Events: Bernoulli(p_event) during wake to create event_times for H_T/H_ST.
    """
    rng = random.Random(seed)
    g = G0.copy()
    D, A = make_full_params(g.number_of_nodes())

    H_S, H_T, H_ST = [], [], []
    event_times = []
    obs_stub = dict(norm=0, MB=0, net=0, tok=0)

    for t in range(steps):
        if t < start_rest:
            phase = "wake"
        else:
            phase = "wake" if (t - start_rest) % rest_every < rest_every - 1 else "rest"

        # ---- WAKE ----
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

            if rng.random() < p_event:
                event_times.append(t)

        # ---- REST ----
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
                g = tmp

        # log entropies
        H_S.append(spatial_entropy(g))
        H_T.append(temporal_entropy(event_times, window=gap_window))
        H_ST.append(spatio_temporal_entropy(g, event_times, window=gap_window))

    return np.array(H_S), np.array(H_T), np.array(H_ST)

def load_x0_graph(pt_path: Path) -> nx.Graph:
    """Load adjacency tensor from .pt and return as NetworkX graph."""
    import torch
    torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
    adj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    elif isinstance(adj, dict) and "adj" in adj:
        adj = adj["adj"]
    adj = (np.array(adj) > 0).astype(int)
    return nx.from_numpy_array(adj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--start_rest_wakeonly", type=int, default=2000, help="Start rest step for wake-only run (>=steps means never)")
    ap.add_argument("--start_rest_wakerest", type=int, default=0, help="Start rest step for wake+rest run")

    ap.add_argument("--rest_every", type=int, default=3)
    ap.add_argument("--rest_replays", type=int, default=5)
    ap.add_argument("--add_edges", type=int, default=2)
    ap.add_argument("--p_event", type=float, default=0.3)
    ap.add_argument("--gap_window", type=int, default=20)

    ap.add_argument("--use_x0", action="store_true")
    ap.add_argument("--pt_path", type=str, default="data/x0.pt")
    ap.add_argument("--out_name", type=str, default="spatiotemporal_entropy_wake_vs_rest.png")
    args = ap.parse_args()

    # Seed graph
    if args.use_x0:
        pt = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
        G_seed = load_x0_graph(pt)
    else:
        N_nodes = 32
        G_seed = nx.watts_strogatz_graph(N_nodes, k=4, p=0.1, seed=42)

    # Run two conditions
    HS_wake, HT_wake, HST_wake = run_schedule(
        G_seed,
        steps=args.steps,
        start_rest=args.start_rest_wakeonly,  # >= steps => never rests
        rest_every=args.rest_every,
        rest_replays=args.rest_replays,
        add_edges=args.add_edges,
        p_event=args.p_event,
        seed=args.seed,
        gap_window=args.gap_window,
    )

    HS_rest, HT_rest, HST_rest = run_schedule(
        G_seed,
        steps=args.steps,
        start_rest=args.start_rest_wakerest,  # rest from t=0 by default
        rest_every=args.rest_every,
        rest_replays=args.rest_replays,
        add_edges=args.add_edges,
        p_event=args.p_event,
        seed=args.seed,
        gap_window=args.gap_window,
    )

    # Plot spatio-temporal entropy only (Figure 9b-style)
    plt.figure(figsize=(7, 4))
    plt.plot(HST_wake, "r--", label="Wake only")
    plt.plot(HST_rest, "b", label="Wake + Rest")
    plt.xlabel("Network Rewiring Steps", fontsize=11)
    plt.ylabel("Spatio-temporal entropy", fontsize=11)
    plt.ylim(0, 4)
    plt.xlim(0, args.steps)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = FIG_DIR / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
