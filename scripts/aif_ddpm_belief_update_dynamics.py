# ==============================================================
# AIF vs. DDPM — belief-update magnitude (KL divergence)
#
# Computes the mean belief-update magnitude Δ_KL(s) over multiple
# runs, highlighting discrete “aha” moments in Active Inference
# versus early single-peak learning in DDPM.
#

# ==============================================================

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import os, numpy as np, torch, networkx as nx, matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# 1.  Posterior helper  (degree‑based ‘net’ factor)
# ──────────────────────────────────────────────────────────────
def posterior_Q(graph, priors):
    """Return posterior dict for factors: norm, MB, tok, net."""
    Q = {f: priors[f].copy() for f in ('norm', 'MB', 'tok')}
    deg = np.array([d for _, d in graph.degree()], dtype=float)
    Q['net'] = deg / deg.sum() if deg.sum() else np.ones_like(deg) / len(deg)
    return Q

PRI = {'norm': np.array([.5, .5]),
       'MB'  : np.array([.5, .5]),
       'tok' : np.array([.5, .5])}

# ──────────────────────────────────────────────────────────────
# 2.  Safe tensor loader (adds NumPy helper to globals)
# ──────────────────────────────────────────────────────────────
def load_adj(pt_path):
    torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
    adj = torch.load(pt_path, weights_only=False)
    return adj.numpy() if isinstance(adj, torch.Tensor) else adj

# ──────────────────────────────────────────────────────────────
# 3.  Gather .pt paths inside a folder (or its single sub‑folder)
# ──────────────────────────────────────────────────────────────
def all_pt_paths(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith('.pt')]
    if files:
        return sorted(files)
    subs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    if len(subs) == 1:
        subf = os.path.join(folder, subs[0])
        return sorted(os.path.join(subf, f) for f in os.listdir(subf)
                      if f.lower().endswith('.pt'))
    return []

def load_Q_sequence(folder):
    paths = all_pt_paths(folder)
    if not paths:
        raise RuntimeError(f'No .pt tensors found in {folder}')
    Qs = []
    for p in paths:
        mat = load_adj(p)
        G   = nx.from_numpy_array(mat)
        Qs.append(posterior_Q(G, PRI))
    return Qs

# ──────────────────────────────────────────────────────────────
# 4.  Tensor folders (under /content)
# ──────────────────────────────────────────────────────────────
dir_aif  = ROOT / "data" / "AIF" / "tensors"
dir_ddpm = ROOT / "data" / "DDPM" / "tensors"           # works if tensors are in DDPM/diffusion/ too

for d in (dir_aif, dir_ddpm):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Folder not found: {d}")

Qs_a = load_Q_sequence(dir_aif)
Qs_d = load_Q_sequence(dir_ddpm)

# ──────────────────────────────────────────────────────────────
# 5.  KL divergence between successive posteriors
# ──────────────────────────────────────────────────────────────
def kl_sequence(Qs, eps=1e-16):
    kl = np.zeros(len(Qs))
    for s in range(1, len(Qs)):
        val = 0.0
        for f in Qs[s]:
            q_new, q_old = Qs[s][f], Qs[s-1][f]
            val += np.sum(q_new * np.log((q_new + eps) / (q_old + eps)))
        kl[s] = val
    return kl

KL_a = kl_sequence(Qs_a)
KL_d = kl_sequence(Qs_d)

# sample indices correspond to 0,100,…,4000
steps = np.arange(len(KL_a)) * 100

# ──────────────────────────────────────────────────────────────
# 6.  Plot with requested axis limits
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(steps, KL_a, lw=2, color='blue', label='AIF')
plt.plot(steps, KL_d, '--', lw=2, color='orange', label='DDPM')

plt.xlabel('Network Reconfiguration Step')
plt.ylabel(r'Belief‑update Magnitude')
plt.title('AIF vs. DDPM: per‑step KL divergence of posteriors')

plt.xlim(0, 4000)
plt.xticks(np.arange(0, 4001, 500))
plt.ylim(0, 0.16)

plt.grid(ls='--', alpha=.4)
plt.legend()
plt.tight_layout()
plt.show()
