# ==============================================================
# Figure 8 — Confidence of multi-cycle BMR with guided pruning
#
# Wake-only baseline:
#   - each step: add K random edges
#
# Wake–Wake–Sleep (multi-cycle):
#   - wake steps: add K random edges
#   - sleep step: MAP-reset then prune up to P edges that
#       (i) connect "weak" nodes (lowest q_net percentile)
#      (ii) satisfy ΔF <= threshold (paper-style Dirichlet test)
#
# Outputs: figure saved to figures/
#
# Example:
#   python scripts/confidence_multicycle_sleep_guided_pruning.py ^
#     --pt_path data/x0.pt --out_dir figures --steps 4000
# ==============================================================

import argparse
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from scipy.special import gammaln


# ------------------------- math helpers -------------------------
def log_beta(a):
    a = a.astype(float)
    return np.sum(gammaln(a)) - gammaln(np.sum(a))

def deltaF(a_post, a_prior, a_reduced):
    # paper-style analytic ΔF expression
    return (log_beta(a_post) + log_beta(a_reduced)
            - log_beta(a_prior) - log_beta(a_post + a_reduced - a_prior))

def entropy(p, eps=1e-32):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0)
    return float(-np.sum(p * np.log(p)))


# ------------------------- IO -------------------------
def load_adjacency_tensor(path: Path):
    torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
    return torch.load(path, weights_only=False)

def to_graph(obj) -> nx.Graph:
    arr = obj.numpy() if isinstance(obj, torch.Tensor) else np.array(obj)
    return nx.from_numpy_array(arr)


# ------------------------- model bits -------------------------
def compute_posterior_Q(graph: nx.Graph, D):
    """Lightweight posterior proxy; q_net proportional to degree."""
    Q = {k: (d / d.sum()) for k, d in D.items()}
    deg = np.array([d for _, d in graph.degree()], dtype=float)
    Q["net"] = deg / deg.sum() if deg.sum() > 0 else np.ones_like(deg) / len(deg)
    return Q

def make_full_params(N):
    # matches your script defaults (can be adjusted if needed)
    D = dict(
        norm=np.array([0.6, 0.4], dtype=float),
        MB=np.array([0.6, 0.4], dtype=float),
        net=np.ones(N, dtype=float) / N,
        tok=np.array([0.4, 0.6], dtype=float),
    )
    A = dict(
        norm=np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float),
        MB=np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float),
        net=np.eye(N, dtype=float),
        tok=np.array([[0.85, 0.15], [0.15, 0.85]], dtype=float),
    )
    return D, A


# ------------------------- ΔF bookkeeping -------------------------
class BMRContext:
    """Dirichlet counts per unordered edge (u,v) for analytic ΔF pruning."""
    def __init__(self, n, alpha0=1.0):
        m = n * (n - 1) // 2
        self.idx = {(u, v): k for k, (u, v) in enumerate(
            (i, j) for i in range(n) for j in range(i + 1, n)
        )}
        self.a_pr = np.full(m, alpha0, dtype=float)
        self.a_po = self.a_pr.copy()

    def _i(self, u, v):
        return self.idx[(u, v) if u < v else (v, u)]

    def edge_added(self, u, v):
        self.a_po[self._i(u, v)] += 1.0

    def map_reset(self):
        self.a_pr[:] = self.a_po

    def dF_cut(self, u, v):
        mask = np.zeros_like(self.a_po)
        mask[self._i(u, v)] = 1.0
        return float(deltaF(self.a_po, self.a_pr, mask))

    def cut(self, u, v):
        i = self._i(u, v)
        self.a_pr[i] = 0.0
        self.a_po[i] = 0.0


# ------------------------- schedules -------------------------
def run_wake_only(g0, D_init, *, steps, log_every, add_per_step):
    g = g0.copy()
    D = deepcopy(D_init)
    tC, Cvals = [], []

    for t in range(steps):
        if t % log_every == 0:
            Q = compute_posterior_Q(g, D)
            tC.append(t)
            Cvals.append(-entropy(Q["net"]))

        ne = list(nx.non_edges(g))
        for _ in range(add_per_step):
            if not ne:
                break
            e = random.choice(ne)
            g.add_edge(*e)
            ne.remove(e)

    return np.asarray(tC), np.asarray(Cvals)

def run_wake_wake_sleep(
    g0, D_init, *, steps, log_every, add_per_wake, prune_max, weak_percent, dF_thresh
):
    g = g0.copy()
    D = deepcopy(D_init)
    bmr = BMRContext(g.number_of_nodes())

    tC, Cvals = [], []

    for t in range(steps):
        phase = t % 3  # 0,1 = wake ; 2 = sleep

        # --- wake: add edges
        if phase < 2:
            ne = list(nx.non_edges(g))
            for _ in range(add_per_wake):
                if not ne:
                    break
                e = random.choice(ne)
                g.add_edge(*e)
                bmr.edge_added(*e)
                ne.remove(e)

        # --- sleep: MAP reset + guided pruning
        else:
            Q = compute_posterior_Q(g, D)

            # MAP reset of priors to current posterior proxy
            for f in D:
                D[f] = Q[f].copy()
            bmr.map_reset()

            # weakest nodes (lowest q_net percentile)
            weak_n = max(1, int(weak_percent * len(Q["net"])))
            weak = set(np.argsort(Q["net"])[:weak_n])

            removed = 0
            # prune edges only inside weak-weak subgraph
            for (u, v) in list(g.edges()):
                if removed >= prune_max:
                    break
                if (u in weak) and (v in weak):
                    if bmr.dF_cut(u, v) <= dF_thresh:
                        g.remove_edge(u, v)
                        bmr.cut(u, v)
                        removed += 1

        # log confidence
        if t % log_every == 0:
            Q = compute_posterior_Q(g, D)
            tC.append(t)
            Cvals.append(-entropy(Q["net"]))

    return np.asarray(tC), np.asarray(Cvals)


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", type=str, default="data/x0.pt", help="Path to x0.pt adjacency")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--add_per_step", type=int, default=3)

    ap.add_argument("--prune_max", type=int, default=10)
    ap.add_argument("--weak_percent", type=float, default=0.20)
    ap.add_argument("--dF_thresh", type=float, default=-3.0)

    ap.add_argument("--plot_xmax", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--out_name", type=str, default="fig8_confidence_multicycle_sleep.png")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ROOT = Path(__file__).resolve().parents[1]
    pt_path = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    tensor = load_adjacency_tensor(pt_path)
    G0 = to_graph(tensor)

    D_full, _ = make_full_params(G0.number_of_nodes())

    # baseline
    random.seed(args.seed)
    t_base, C_base = run_wake_only(
        G0, D_full, steps=args.steps, log_every=args.log_every, add_per_step=args.add_per_step
    )

    # multi-cycle
    random.seed(args.seed)
    t_ext, C_ext = run_wake_wake_sleep(
        G0, D_full,
        steps=args.steps,
        log_every=args.log_every,
        add_per_wake=args.add_per_step,
        prune_max=args.prune_max,
        weak_percent=args.weak_percent,
        dF_thresh=args.dF_thresh,
    )

    # plot
    plt.figure(figsize=(9, 3))
    plt.plot(t_base, C_base, "r--", label="Wake only")
    plt.plot(t_ext, C_ext, "b-", label="Multi-cycle sleep")

    # optional vertical markers for sleep steps
    for s in range(0, args.steps, 3):
        if s % 3 == 2:
            plt.axvline(s, color="k", ls=":", lw=0.3)

    plt.ylabel("Confidence")
    plt.xlabel("Network Rewiring Steps")
    plt.xlim(0, args.plot_xmax)
    plt.grid(True, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()

    out_path = out_dir / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
