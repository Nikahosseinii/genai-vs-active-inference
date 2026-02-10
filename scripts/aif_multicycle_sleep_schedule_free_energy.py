# ==============================================================
# Active Inference — wake-only vs multi-cycle wake–wake–sleep schedule
#
# Wake phase: greedy free-energy edge additions (3 edges per step).
# Sleep phase: MAP reset + structural pruning of weak-node edges
#              using ΔF threshold (and cap on deletions).
#
# Intended for Figure 7(b)-style plot in the paper.
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

# ---------- analytic ΔF -----------------------------------------
def log_beta(a):
    a = np.asarray(a, dtype=float) + 1e-16
    return np.sum(gammaln(a)) - gammaln(np.sum(a))

def deltaF(a_post, a_prior, a_red):
    return (log_beta(a_post) + log_beta(a_red)
            - log_beta(a_prior) - log_beta(a_post + a_red - a_prior))

# ---------- helpers ---------------------------------------------
def free_energy(Q, D, A, obs, eps=1e-32):
    F = 0.0
    for f in ["norm", "MB", "net", "tok"]:
        q, d = Q[f], D[f]
        F += np.sum(q * (np.log(q + eps) - np.log(d + eps)))    # complexity
        F -= np.sum(q * np.log(A[f][obs[f]] + eps))             # accuracy
    return float(F)

def posterior_Q(g, D):
    Q = {k: d / d.sum() for k, d in D.items()}
    deg = np.array([d for _, d in g.degree()], float)
    Q["net"] = deg / deg.sum() if deg.sum() else np.ones_like(deg) / len(deg)
    return Q

def load_adjacency_tensor(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    return np.array(obj)

# ---------- Dirichlet bookkeeping -------------------------------
class BMR:
    def __init__(self, n, alpha0=1.0):
        m = n * (n - 1) // 2
        self.map = {(u, v): k for k, (u, v) in enumerate(
            (i, j) for i in range(n) for j in range(i + 1, n)
        )}
        self.a_pr = np.full(m, alpha0, dtype=float)
        self.a_po = self.a_pr.copy()

    def _idx(self, u, v):
        return self.map[(u, v) if u < v else (v, u)]

    def add(self, u, v):
        self.a_po[self._idx(u, v)] += 1.0

    def reset(self):
        self.a_pr[:] = self.a_po

    def DF(self, u, v):
        mask = np.zeros_like(self.a_po)
        mask[self._idx(u, v)] = 1.0
        return float(deltaF(self.a_po, self.a_pr, mask))

    def cut(self, u, v):
        i = self._idx(u, v)
        self.a_pr[i] = self.a_po[i] = 0.0

# ---------- priors & likelihoods --------------------------------
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

# ---------- scheduler -------------------------------------------
def run(G0, D0, A0, *, steps=2100, log_every=50, n_cand=15,
        start_sleep=500, deltaF_threshold=-3.0, weak_frac=0.2,
        max_deletions=10, edges_per_wake_step=3):
    """Run rewiring with wake–wake–sleep cycle starting at `start_sleep`."""
    g = G0.copy()
    D, A = deepcopy(D0), deepcopy(A0)
    bmr = BMR(g.number_of_nodes())
    obs = dict(norm=0, MB=0, tok=0, net=0)
    tF, F = [], []

    for t in range(steps):
        # decide phase
        if t < start_sleep:
            phase = "wake"
        else:
            phase = "wake" if (t - start_sleep) % 3 < 2 else "sleep"

        if phase == "wake":
            obs.update(dict(
                norm=random.randint(0, 1),
                MB=random.randint(0, 1),
                tok=random.randint(0, 1),
                net=random.randrange(g.number_of_nodes()),
            ))

            ne = list(nx.non_edges(g))
            for _ in range(edges_per_wake_step):
                if not ne:
                    break
                cand = random.sample(ne, min(n_cand, len(ne)))
                Q0 = posterior_Q(g, D)
                F0 = free_energy(Q0, D, A, obs)
                best, best_delta = None, 0.0
                for u, v in cand:
                    g.add_edge(u, v)
                    delta = free_energy(posterior_Q(g, D), D, A, obs) - F0
                    if delta < best_delta:
                        best, best_delta = (u, v), delta
                    g.remove_edge(u, v)
                if best:
                    g.add_edge(*best)
                    bmr.add(*best)
                    if best in ne:
                        ne.remove(best)

        else:  # sleep
            Q = posterior_Q(g, D)
            k = max(1, int(weak_frac * len(Q["net"])))
            weak = set(np.argsort(Q["net"])[:k])

            # MAP reset
            for f in D:
                D[f] = Q[f].copy()
            bmr.reset()

            removed = 0
            candidate_edges = [e for e in g.edges() if e[0] in weak and e[1] in weak]
            for u, v in candidate_edges:
                if removed >= max_deletions:
                    break
                if bmr.DF(u, v) <= deltaF_threshold:
                    g.remove_edge(u, v)
                    bmr.cut(u, v)
                    removed += 1

        if t % log_every == 0:
            Q = posterior_Q(g, D)
            tF.append(t)
            F.append(free_energy(Q, D, A, obs))

    return np.array(tF), np.array(F)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", type=str, default="data/x0.pt")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2100)
    ap.add_argument("--start_sleep", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    pt_path = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    adj = load_adjacency_tensor(pt_path)
    G0 = nx.from_numpy_array(adj)
    D0, A0 = make_full_params(G0.number_of_nodes())

    # baseline (wake-only): set start_sleep beyond steps so it never sleeps
    random.seed(args.seed)
    t_base, F_base = run(G0, D0, A0, steps=args.steps, start_sleep=args.steps + 1, log_every=args.log_every)

    # multi-cycle wake–wake–sleep
    random.seed(args.seed)
    t_blue, F_blue = run(G0, D0, A0, steps=args.steps, start_sleep=args.start_sleep, log_every=args.log_every)

    # plot
    plt.figure(figsize=(8, 4))
    plt.plot(t_base, F_base, "r--", label="Wake only")
    plt.plot(t_blue, F_blue, "b", label=f"Multi-cycle sleep starts at step {args.start_sleep}")

    # dotted lines on sleep steps
    for s in range(args.start_sleep, args.steps, 3):
        if (s - args.start_sleep) % 3 == 2:
            plt.axvline(s, color="k", ls=":", lw=0.4)

    plt.xlabel("Network Rewiring Steps", fontsize=12)
    plt.ylabel("Free energy (Nats)", fontsize=12)
    plt.xlim(0, 2000)
    plt.legend(frameon=False)
    plt.grid(True)
    plt.tight_layout()

    # sleep band label near start
    bar_width = 30
    plt.axvspan(args.start_sleep, args.start_sleep + bar_width, color=(0.4, 0.6, 1.0, 0.2), zorder=0)
    y_text = max(np.max(F_base), np.max(F_blue)) - 10.35
    plt.text(args.start_sleep + bar_width / 2, y_text, "Sleep",
             rotation=90, fontsize=12, color="blue", ha="center", va="bottom")

    out_path = out_dir / "free_energy_multicycle_wake_wake_sleep.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
