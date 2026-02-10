# ==============================================================
# Figure 11(c) — Prior entropy under four strategies (deterministic)
#
# Strategies:
#   1) Wake-only (FE-minimising edge additions)
#   2) Sleep (wake-wake-sleep + BMR pruning)
#   3) Rest  (wake-wake-rest + entropy-max replays)
#   4) Psychedelics (precision drop + random adds in window)
#
# Metric: prior entropy H(D_t) for a 2-state categorical prior D.
#
# Example:
#   python scripts/prior_entropy_strategies_fig11c.py ^
#     --pt_path data/x0.pt --out_dir figures --steps 4000 ^
#     --psy_start 500 --psy_end 1000 --seed 3
# ==============================================================

import argparse
import random
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
from scipy.special import gammaln


# ------------------------- IO -------------------------
def load_graph(path: Path) -> nx.Graph:
    torch.serialization.add_safe_globals(["numpy._core.multiarray._reconstruct"])
    adj = torch.load(path, weights_only=False)
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    elif isinstance(adj, dict) and "adj" in adj:
        adj = adj["adj"]
    return nx.from_numpy_array((adj > 0).astype(int))


# ------------------------- math -------------------------
def cat_entropy(p):
    p = np.asarray(p, dtype=float)
    return float(-np.sum(p * np.log(p + 1e-32)))

def log_beta(a):
    a = np.asarray(a, dtype=float) + 1e-16
    return np.sum(gammaln(a)) - gammaln(np.sum(a))

def deltaF(a_post, a_prior, a_red):
    # paper-style analytic ΔF for Dirichlet reduction
    return (log_beta(a_post) + log_beta(a_red)
            - log_beta(a_prior) - log_beta(a_post + a_red - a_prior))


# ------------------------- core simulator -------------------------
def run_sim(
    G0: nx.Graph,
    *,
    steps: int,
    seed: int,
    mode: str,
    # precision + learning
    w_high: float,
    w_low: float,
    eta: float,
    # psychedelic window
    psy_start: int,
    psy_end: int,
    # sleep/rest knobs
    weak_percent: float,
    prune_max: int,
    dF_thresh: float,
    rest_replays: int,
):
    # full determinism
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    G = G0.copy()
    N = G.number_of_nodes()

    # 2-state generative prior and likelihood (matches your spirit; tweak if needed)
    D = np.array([0.4, 0.6], dtype=float)
    A = np.array([[0.9, 0.1],
                  [0.2, 0.8]], dtype=float)
    obs = 1  # fixed observation

    # Dirichlet bookkeeping over all unordered edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    idx_map = {e: k for k, e in enumerate(all_edges)}
    a_prior = np.ones(len(all_edges), dtype=float)
    a_post  = a_prior.copy()

    non_edges = list(nx.non_edges(G))

    H = np.zeros(steps, dtype=float)

    for t in range(steps):
        # precision schedule (psychedelic window: low precision)
        if t < psy_start:
            w = w_high
        elif t < psy_end:
            w = w_low
        else:
            # smooth recovery to high precision
            denom = max(1, (steps - psy_end))
            w = w_low + (w_high - w_low) * (t - psy_end) / denom

        # posterior sampling from Dirichlet(w * D)
        alpha = w * D
        Qs = np.vstack([np_rng.dirichlet(alpha) for _ in range(N)])
        Qbar = Qs.mean(axis=0)

        # prior update + entropy log
        D = (1 - eta) * D + eta * Qbar
        D = D / D.sum()
        H[t] = cat_entropy(D)

        # ------------------ rewiring ------------------
        # Psychedelics: inside window do random additions (no FE test, no pruning)
        if mode == "psy" and (psy_start <= t < psy_end):
            if non_edges:
                e = rng.choice(non_edges)
                G.add_edge(*e)
                non_edges.remove(e)
                # treat as evidence (optional): increment corresponding posterior count
                e_norm = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                a_post[idx_map[e_norm]] += 1.0
            continue

        # phase selection for sleep/rest schedules
        if mode == "baseline":
            phase = "wake"
        elif mode == "sleep":
            phase = "sleep" if (t % 3 == 2) else "wake"
        elif mode == "rest":
            phase = "rest" if (t % 3 == 2) else "wake"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- WAKE: FE-minimising add (pick best among 3 candidates) ----
        if phase == "wake" and non_edges:
            cand = rng.sample(non_edges, min(3, len(non_edges)))
            best_e, best_val = None, float("inf")

            # simple FE proxy for single factor
            # F(Q,D,A,o) = KL(Q||D) - E_Q[log A[o]]
            def F_single(Q, D, A, o):
                return float(np.sum(Q * (np.log(Q + 1e-32) - np.log(D + 1e-32)))
                             - np.sum(Q * np.log(A[o] + 1e-32)))

            for e in cand:
                G.add_edge(*e)
                val = F_single(Qbar, D, A, obs)
                if val < best_val:
                    best_e, best_val = e, val
                G.remove_edge(*e)

            if best_e is not None:
                G.add_edge(*best_e)
                non_edges.remove(best_e)
                e_norm = (best_e[0], best_e[1]) if best_e[0] < best_e[1] else (best_e[1], best_e[0])
                a_post[idx_map[e_norm]] += 1.0

        # ---- SLEEP: MAP reset + guided pruning (weak nodes + ΔF test) ----
        elif phase == "sleep":
            # MAP reset: prior becomes posterior counts (paper-style)
            a_prior[:] = a_post

            # weak nodes by degree fraction
            deg = np.array([d for _, d in G.degree()], dtype=float)
            if deg.sum() > 0:
                q_net = deg / deg.sum()
                weak_n = max(1, int(weak_percent * N))
                weak = set(np.argsort(q_net)[:weak_n])

                removed = 0
                for (u, v) in list(G.edges()):
                    if removed >= prune_max:
                        break
                    if (u in weak) and (v in weak):
                        e_norm = (u, v) if u < v else (v, u)
                        idx = idx_map[e_norm]
                        a_red = np.zeros_like(a_post)
                        a_red[idx] = 1.0

                        if deltaF(a_post, a_prior, a_red) <= dF_thresh:
                            G.remove_edge(u, v)
                            non_edges.append(e_norm)
                            # zero out that edge in both prior/post to represent a cut
                            a_post[idx] = 0.0
                            a_prior[idx] = 0.0
                            removed += 1

        # ---- REST: entropy-ish fictive replays (random add/remove) ----
        elif phase == "rest":
            for _ in range(rest_replays):
                tmp = G.copy()
                if rng.random() < 0.5 and tmp.number_of_edges() > 0:
                    e = rng.choice(list(tmp.edges()))
                    tmp.remove_edge(*e)
                    e_norm = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                    non_edges.append(e_norm)
                else:
                    ne = list(nx.non_edges(tmp))
                    if ne:
                        e = rng.choice(ne)
                        tmp.add_edge(*e)
                        e_norm = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
                        # optional: count as evidence
                        a_post[idx_map[e_norm]] += 1.0
                        if e_norm in non_edges:
                            non_edges.remove(e_norm)
                G = tmp

    return H


# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", type=str, default="data/x0.pt")
    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--out_name", type=str, default="fig11c_prior_entropy_strategies.png")

    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=3)

    ap.add_argument("--psy_start", type=int, default=500)
    ap.add_argument("--psy_end", type=int, default=1000)

    ap.add_argument("--w_high", type=float, default=12.0)
    ap.add_argument("--w_low", type=float, default=1.5)
    ap.add_argument("--eta", type=float, default=0.05)

    ap.add_argument("--weak_percent", type=float, default=0.20)
    ap.add_argument("--prune_max", type=int, default=10)
    ap.add_argument("--dF_thresh", type=float, default=-3.0)
    ap.add_argument("--rest_replays", type=int, default=5)

    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    pt_path = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    G_seed = load_graph(pt_path)

    modes = {
        "Wake": "baseline",
        "Sleep": "sleep",
        "Rest": "rest",
        "Psychedelics": "psy",
    }

    curves = {}
    for name, mode in modes.items():
        curves[name] = run_sim(
            G_seed,
            steps=args.steps,
            seed=args.seed,
            mode=mode,
            w_high=args.w_high,
            w_low=args.w_low,
            eta=args.eta,
            psy_start=args.psy_start,
            psy_end=args.psy_end,
            weak_percent=args.weak_percent,
            prune_max=args.prune_max,
            dF_thresh=args.dF_thresh,
            rest_replays=args.rest_replays,
        )

    # plot
    x = np.arange(args.steps)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    linestyles = ["-", "--", "-.", ":"]

    plt.figure(figsize=(9, 3.5))
    for (name, y), c, ls in zip(curves.items(), palette, linestyles):
        plt.plot(x, y, color=c, lw=2, linestyle=ls, label=name)

    plt.axvspan(args.psy_start, args.psy_end, color="peachpuff", alpha=0.3, label="Psychedelic window")
    plt.xlabel("Network Reconfiguration Steps", fontsize=12)
    plt.ylabel("Prior Entropy", fontsize=12)
    plt.xlim(0, args.steps)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / args.out_name
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
