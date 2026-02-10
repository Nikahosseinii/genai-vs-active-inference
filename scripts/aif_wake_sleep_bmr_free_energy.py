# ==============================================================
# Active Inference — wake-only vs wake+sleep (BMR pruning)
#
# Reproduces the free-energy trajectory comparison where a single
# sleep block triggers Bayesian Model Reduction (BMR) pruning of edges
# (ΔF threshold), after greedy-F wake rewiring.
#
# Intended for Figure 7(a)-style plot in the paper.
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

plt.rcParams.update({"font.size": 10})

# ───────── helper: log-beta & analytic ΔF (paper’s Eq. 4.1) ────
def log_beta(a):
    a = a.astype(float)
    return np.sum(gammaln(a)) - gammaln(np.sum(a))

def deltaF(a_post, a_prior, a_reduced):
    return (log_beta(a_post) + log_beta(a_reduced)
            - log_beta(a_prior) - log_beta(a_post + a_reduced - a_prior))

# ───────── misc helpers ────────────────────────────────────────
def load_adjacency_tensor(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return np.array(obj)

def entropy(p, eps=1e-32):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def compute_free_energy(Q, D, A, obs, eps=1e-32):
    F = 0.0
    for f in ["norm", "MB", "net", "tok"]:
        q, d = Q[f], D[f]
        F += np.sum(q * (np.log(q + eps) - np.log(d + eps)))   # complexity
        F -= np.sum(q * np.log(A[f][obs[f]] + eps))            # accuracy
    return float(F)

def compute_posterior_Q(graph, D):
    Q = {k: d / d.sum() for k, d in D.items()}
    deg = np.array([d for _, d in graph.degree()], float)
    Q["net"] = deg / deg.sum() if deg.sum() else np.ones_like(deg) / len(deg)
    return Q

# ───────── Dirichlet bookkeeping for ΔF tests ──────────────────
class BMRContext:
    def __init__(self, n, alpha0=1.0):
        self.size = n * (n - 1) // 2
        self.idxmap = {
            (u, v): k
            for k, (u, v) in enumerate((i, j) for i in range(n) for j in range(i + 1, n))
        }
        self.a_pr = np.full(self.size, alpha0, float)  # priors
        self.a_po = self.a_pr.copy()                   # posteriors

    def _i(self, u, v):
        return self.idxmap[(u, v) if u < v else (v, u)]

    def edge_added(self, u, v):
        self.a_po[self._i(u, v)] += 1

    def map_reset(self):
        self.a_pr[:] = self.a_po

    def deltaF_cut(self, u, v):
        mask = np.zeros_like(self.a_po)
        mask[self._i(u, v)] = 1
        return float(deltaF(self.a_po, self.a_pr, mask))

    def cut(self, u, v):
        i = self._i(u, v)
        self.a_pr[i] = self.a_po[i] = 0.0

# ───────── priors & likelihoods ────────────────────────────────
def make_full_params(N):
    pointy = np.full(N, 0.4 / (N - 1))
    pointy[0] = 0.6
    D = dict(
        norm=np.array([0.4, 0.6]),
        MB=np.array([0.4, 0.6]),
        net=pointy,
        tok=np.array([0.4, 0.6]),
    )
    off = 0.1 / (N - 1) if N > 1 else 0
    net_like = np.full((N, N), off)
    np.fill_diagonal(net_like, 0.9)
    A = dict(
        norm=np.array([[0.9, 0.1], [0.1, 0.9]]),
        MB=np.array([[0.8, 0.2], [0.2, 0.8]]),
        net=net_like,
        tok=np.array([[0.85, 0.15], [0.15, 0.85]]),
    )
    return D, A

# ───────── scheduler ───────────────────────────────────────────
def run_schedule(G0, D0, A0, *, trials, steps_per_trial,
                 sleep_steps=(), logF=50, logC=50, n_cand=10,
                 deltaF_threshold=-3.0):
    sleep_steps = set(sleep_steps)
    g = G0.copy()
    D = deepcopy(D0)
    A = deepcopy(A0)
    bmr = BMRContext(g.number_of_nodes())
    obs = {"norm": 0, "MB": 0, "tok": 0, "net": 0}
    tF, F, tC, C = [], [], [], []
    t = 0

    def sleep_block():
        nonlocal t
        # log once so red and blue share the same value at the sleep step
        Q = compute_posterior_Q(g, D)
        tF.append(t)
        F.append(compute_free_energy(Q, D, A, obs))
        tC.append(t)
        C.append(-entropy(Q["net"]))

        # 1) MAP reset
        for f in D:
            D[f] = Q[f].copy()
        bmr.map_reset()

        # 2) evaluate ΔF for every existing edge; cut if ΔF <= threshold
        for (u, v) in list(g.edges()):
            if bmr.deltaF_cut(u, v) <= deltaF_threshold:
                g.remove_edge(u, v)
                bmr.cut(u, v)

    for _ in range(trials):
        for _ in range(steps_per_trial):
            obs.update(
                dict(
                    norm=random.randint(0, 1),
                    MB=random.randint(0, 1),
                    tok=random.randint(0, 1),
                    net=random.randrange(g.number_of_nodes()),
                )
            )

            if t % logF == 0 and t not in sleep_steps:
                Q = compute_posterior_Q(g, D)
                tF.append(t)
                F.append(compute_free_energy(Q, D, A, obs))

            if t % logC == 0:
                Q = compute_posterior_Q(g, D)
                tC.append(t)
                C.append(-entropy(Q["net"]))

            # greedy edge addition
            ne = list(nx.non_edges(g))
            if ne:
                cand = random.sample(ne, min(n_cand, len(ne)))
                Q0 = compute_posterior_Q(g, D)
                F0 = compute_free_energy(Q0, D, A, obs)
                best, best_delta = None, 0.0
                for i, j in cand:
                    g.add_edge(i, j)
                    delta = compute_free_energy(compute_posterior_Q(g, D), D, A, obs) - F0
                    if delta < best_delta:
                        best, best_delta = (i, j), delta
                    g.remove_edge(i, j)
                if best:
                    g.add_edge(*best)
                    bmr.edge_added(*best)

            t += 1
            if t in sleep_steps:
                sleep_block()
                sleep_steps.remove(t)

    return np.array(tF), np.array(F), np.array(tC), np.array(C)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", type=str, default="data/x0.pt", help="Path to x0.pt adjacency tensor")
    ap.add_argument("--out_dir", type=str, default="figures", help="Where to save the plot")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--total_steps", type=int, default=4000)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--sleep_at", type=int, default=500)
    ap.add_argument("--logF", type=int, default=50)
    ap.add_argument("--deltaF_threshold", type=float, default=-3.0)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    pt_path = (ROOT / args.pt_path).resolve() if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    adj = load_adjacency_tensor(pt_path)
    G0 = nx.from_numpy_array(adj)
    D0, A0 = make_full_params(G0.number_of_nodes())

    steps_per_trial = args.total_steps // args.trials

    random.seed(args.seed)
    tF_r, F_r, _, _ = run_schedule(
        G0, D0, A0, trials=args.trials, steps_per_trial=steps_per_trial,
        sleep_steps=(), logF=args.logF, deltaF_threshold=args.deltaF_threshold
    )

    random.seed(args.seed)
    tF_s, F_s, _, _ = run_schedule(
        G0, D0, A0, trials=args.trials, steps_per_trial=steps_per_trial,
        sleep_steps=(args.sleep_at,), logF=args.logF, deltaF_threshold=args.deltaF_threshold
    )

    # Plot
    plt.figure(figsize=(8, 4))
    mask_r = tF_r >= args.sleep_at
    plt.plot(tF_r[mask_r], F_r[mask_r], "r--", label="Wake only")

    plt.plot(tF_s, F_s, "b", label=f"Single-cycle sleep starts at step {args.sleep_at}")
    plt.axvline(args.sleep_at, color="k", ls=":", lw=0.8)

    # light blue sleep band
    bar_width = 30
    plt.axvspan(args.sleep_at, args.sleep_at + bar_width, color=(0.4, 0.6, 1.0, 0.2), zorder=0)

    y_text = max(np.max(F_r), np.max(F_s)) - 0.1 * abs(max(np.max(F_r), np.max(F_s)))
    plt.text(args.sleep_at + bar_width / 2, y_text, "Sleep", rotation=90,
             fontsize=12, color="blue", ha="center", va="bottom")

    plt.xlim(0, 2000)
    plt.xlabel("Network Rewiring Steps", fontsize=12)
    plt.ylabel("Free energy (Nats)", fontsize=12)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()

    out_path = out_dir / "free_energy_wake_vs_sleep_bmr.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
