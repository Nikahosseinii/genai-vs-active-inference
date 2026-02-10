# ==============================================================
# Figure 11(a,b) — Psychedelic intervention (synthetic demo)
#
# Outputs:
#   (a) Prior entropy of D vs steps (wake-only vs psychedelics)
#   (b) Posterior synchrony vs steps (wake-only vs psychedelics)
#
# Design:
#   - Shared prefix (0..psy_start-1) uses identical RNG stream, so
#     both conditions match pre-window.
#   - Psychedelic window: lower precision + explicit prior broadening
#     (uniform-mix and/or temperature soften).
#   - Post-window: optionally freeze learning to keep entropy high.
#
# Usage:
#   python scripts/psychedelics_prior_entropy_and_synchrony.py --use_x0 --pt_path data/x0.pt
# ==============================================================

import argparse
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ------------------------- utilities -------------------------
def cat_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-32)
    return float(-np.sum(p * np.log(p + 1e-32)))

def tv_distance(p, q):
    return 0.5 * np.abs(p - q).sum()

def mix_with_uniform(p, lam):
    k = p.size
    return (1 - lam) * p + lam * np.ones(k) / k

def soften(p, tau):
    q = np.exp(np.log(p + 1e-32) / tau)
    return q / q.sum()

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

def precision_schedule(t_abs, w_high, w_low, psy_start, psy_end, ramp):
    if ramp == 0:
        return w_low if (psy_start <= t_abs < psy_end) else w_high

    # linear ramps in/out
    if t_abs < psy_start - ramp:
        return w_high
    if psy_start - ramp <= t_abs < psy_start:
        return w_high - (w_high - w_low) * (t_abs - (psy_start - ramp)) / ramp
    if psy_start <= t_abs < psy_end:
        return w_low
    if psy_end <= t_abs < psy_end + ramp:
        return w_low + (w_high - w_low) * (t_abs - psy_end) / ramp
    return w_high

def simulate_block(
    G,
    steps,
    t0,
    *,
    w_fn,
    eta_fn,
    D_init,
    rng_state,
    psy_start,
    psy_end,
    broaden=False,
    lambda_unif=0.6,
    tau_soften=2.0,
):
    """
    Returns:
      ent: prior entropy over time
      syn: posterior synchrony over time
      D_final, rng_state_final
    """
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state

    D = D_init.copy()
    N = G.number_of_nodes()
    nodes = list(G.nodes)

    ent, syn = [], []

    for t_rel in range(steps):
        t_abs = t0 + t_rel
        w = float(w_fn(t_abs))
        eta = float(eta_fn(t_abs))

        # Sample per-node posteriors (Dirichlet around D scaled by precision w)
        alpha = w * D
        posts = np.vstack([rng.dirichlet(alpha) for _ in nodes])

        # Update prior D (learning)
        if eta > 0:
            mean_post = posts.mean(axis=0)

            if broaden and (psy_start <= t_abs < psy_end):
                if lambda_unif > 0:
                    mean_post = mix_with_uniform(mean_post, lambda_unif)
                if tau_soften is not None and tau_soften > 0:
                    mean_post = soften(mean_post, tau_soften)

            D = (1 - eta) * D + eta * mean_post
            D /= D.sum()

        ent.append(cat_entropy(D))

        # Posterior synchrony = average over pairs of (1 - TV distance)
        if N > 1:
            acc = 0.0
            for i in range(N):
                Qi = posts[i]
                for j in range(i + 1, N):
                    acc += 1.0 - tv_distance(Qi, posts[j])
            syn.append(2.0 * acc / (N * (N - 1)))
        else:
            syn.append(1.0)

    return np.array(ent), np.array(syn), D, rng.bit_generator.state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_total", type=int, default=2000)
    ap.add_argument("--psy_start", type=int, default=500)
    ap.add_argument("--psy_end", type=int, default=1000)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--w_high", type=float, default=12.0)
    ap.add_argument("--w_low", type=float, default=1.5)
    ap.add_argument("--ramp", type=int, default=0)

    ap.add_argument("--eta_high", type=float, default=0.02)
    ap.add_argument("--eta_low", type=float, default=0.06)
    ap.add_argument("--eta_post", type=float, default=0.00)

    ap.add_argument("--lambda_unif", type=float, default=0.6)
    ap.add_argument("--tau_soften", type=float, default=2.0)

    ap.add_argument("--use_x0", action="store_true")
    ap.add_argument("--pt_path", type=str, default="data/x0.pt")

    ap.add_argument("--out_dir", type=str, default="figures")
    ap.add_argument("--out_prefix", type=str, default="fig11")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Seed graph (not used directly in the synthetic posterior sampling, but kept for consistency)
    if args.use_x0:
        pt = (ROOT / args.pt_path) if not Path(args.pt_path).is_absolute() else Path(args.pt_path)
        G = load_x0_graph(pt)
    else:
        G = nx.watts_strogatz_graph(32, k=4, p=0.1, seed=1)

    steps_total = args.steps_total
    psy_start = args.psy_start
    psy_end = args.psy_end
    if not (0 <= psy_start < psy_end <= steps_total):
        raise ValueError("Require 0 <= psy_start < psy_end <= steps_total")

    def eta_schedule(t_abs):
        if psy_start <= t_abs < psy_end:
            return args.eta_low
        if t_abs >= psy_end:
            return args.eta_post
        return args.eta_high

    # RNG + initial prior D (binary categorical)
    rng0 = np.random.default_rng(args.seed).bit_generator.state
    D0 = np.array([0.4, 0.6], dtype=float)

    # Shared prefix (0..psy_start-1)
    ent_pref, syn_pref, D_pref, rng_pref = simulate_block(
        G,
        psy_start,
        0,
        w_fn=lambda t: args.w_high,
        eta_fn=eta_schedule,
        D_init=D0,
        rng_state=rng0,
        psy_start=psy_start,
        psy_end=psy_end,
        broaden=False,
        lambda_unif=args.lambda_unif,
        tau_soften=args.tau_soften,
    )

    tail_len = steps_total - psy_start

    # Wake-only tail (always high precision, no broadening)
    ent_w_tail, syn_w_tail, _, _ = simulate_block(
        G,
        tail_len,
        psy_start,
        w_fn=lambda t: args.w_high,
        eta_fn=eta_schedule,
        D_init=D_pref,
        rng_state=rng_pref,
        psy_start=psy_start,
        psy_end=psy_end,
        broaden=False,
        lambda_unif=args.lambda_unif,
        tau_soften=args.tau_soften,
    )

    # Psychedelic tail (precision drop + broadening inside window)
    ent_p_tail, syn_p_tail, _, _ = simulate_block(
        G,
        tail_len,
        psy_start,
        w_fn=lambda t: precision_schedule(t, args.w_high, args.w_low, psy_start, psy_end, args.ramp),
        eta_fn=eta_schedule,
        D_init=D_pref,
        rng_state=rng_pref,
        psy_start=psy_start,
        psy_end=psy_end,
        broaden=True,
        lambda_unif=args.lambda_unif,
        tau_soften=args.tau_soften,
    )

    ent_wake = np.concatenate([ent_pref, ent_w_tail])
    syn_wake = np.concatenate([syn_pref, syn_w_tail])
    ent_psy = np.concatenate([ent_pref, ent_p_tail])
    syn_psy = np.concatenate([syn_pref, syn_p_tail])

    x = np.arange(steps_total)
    band = dict(color="peachpuff", alpha=0.3, zorder=0, label="Psychedelic window")

    # ---- Fig 11(a): prior entropy
    plt.figure(figsize=(8, 4))
    plt.axvspan(psy_start, psy_end, **band)
    plt.plot(x, ent_wake, "r--", label="Wake only")
    plt.plot(x, ent_psy, "b-", label="With psychedelics")
    plt.ylabel("Prior Entropy", fontsize=12)
    plt.xlabel("Network Rewiring Steps", fontsize=12)
    plt.xlim(0, steps_total)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_a = out_dir / f"{args.out_prefix}_a_prior_entropy.png"
    plt.savefig(out_a, dpi=300, bbox_inches="tight")
    print("Saved:", out_a)

    # ---- Fig 11(b): posterior synchrony
    plt.figure(figsize=(8, 4))
    plt.axvspan(psy_start, psy_end, **band)
    plt.plot(x, syn_wake, "r--", label="Wake only")
    plt.plot(x, syn_psy, "b-", label="With psychedelics")
    plt.ylabel("Posterior Synchrony", fontsize=12)
    plt.xlabel("Network Rewiring Steps", fontsize=12)
    plt.xlim(0, steps_total)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_b = out_dir / f"{args.out_prefix}_b_posterior_synchrony.png"
    plt.savefig(out_b, dpi=300, bbox_inches="tight")
    print("Saved:", out_b)


if __name__ == "__main__":
    main()
