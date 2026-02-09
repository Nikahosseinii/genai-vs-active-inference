# ============================================================
#  Full benchmark script: Active-Inference vs. contagion
#  — produces shaded blue / red / pink plots
# ============================================================
import os
import random
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  (1)  Utility functions from the original code
# ------------------------------------------------------------
def load_adjacency_tensor(file_path):
    torch.serialization.add_safe_globals(['numpy._core.multiarray._reconstruct'])
    return torch.load(file_path, weights_only=False)

def compute_free_energy(Q, D, A, y, eps=1e-16):
    F = 0.0
    for f in ['norm', 'MB', 'net', 'tok']:
        q = Q[f]
        d = D[f]
        F += np.sum(q * (np.log(q + eps) - np.log(d + eps)))
        lik = A[f][y[f]]
        F -= np.sum(q * np.log(lik + eps))
    return F

def compute_posterior_Q(graph, observations, priors, likelihoods):
    Q = {f: priors[f].copy() for f in priors}
    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
    Q['net'] = degrees / degrees.sum() if degrees.sum() > 0 else np.ones_like(degrees) / len(degrees)
    return Q

def update_network(network, observations, priors, likelihoods,
                   n_candidates=20, max_additions=3):
    Q_cur = compute_posterior_Q(network, observations, priors, likelihoods)
    F_cur = compute_free_energy(Q_cur, priors, likelihoods, observations)
    best_net, best_F = None, F_cur
    non_edges = list(nx.non_edges(network))

    for _ in range(n_candidates):
        k = random.randint(1, max_additions)
        if len(non_edges) < k:
            continue
        cand = network.copy()
        cand.add_edges_from(random.sample(non_edges, k))
        if not nx.is_connected(cand):
            continue
        Qp = compute_posterior_Q(cand, observations, priors, likelihoods)
        Fp = compute_free_energy(Qp, priors, likelihoods, observations)
        if Fp < best_F:
            best_net, best_F = cand, Fp
    return (best_net, best_F) if best_net else (network, F_cur)

def active_inference_optimization(n_steps, initial_network,
                                  observations, priors, likelihoods,
                                  n_candidates=20, max_additions=3):
    nets, Fs = [initial_network.copy()], []
    net = initial_network.copy()

    for _ in range(n_steps):
        net, Fval = update_network(net, observations, priors, likelihoods,
                                   n_candidates, max_additions)
        nets.append(net.copy())
        Fs.append(Fval)
    return nets, Fs

def spread_disease(graph, start_node, r, return_tokens=False, delay=1):
    infected, time_units, tokens = {start_node}, 0, 0
    while len(infected) < graph.number_of_nodes():
        new_inf = {node for node in graph.nodes() if node not in infected and
                   any(nbr in infected and np.random.rand() < r for nbr in graph.neighbors(node))}
        if not new_inf:
            break
        tokens += 2 * sum(len([nbr for nbr in graph.neighbors(node) if nbr in infected]) for node in new_inf)
        infected |= new_inf
        time_units += 1
    return (time_units * delay, tokens) if return_tokens else time_units * delay

# ------------------------------------------------------------
#  (2)  Experiment set-up
# ------------------------------------------------------------
FILE_PT   = "x0.pt"               # path to adjacency matrix
N_RUNS    = 70                    # Monte-Carlo repetitions
N_STEPS   = 4_000                 # AIF optimisation steps
R_VALS    = [0.5, 0.75, 1.0]      # spreading probabilities
N_CAND    = 20                    # candidate graphs per step
MAX_ADD   = 3                     # max edges to add per candidate

# ---- load base graph once ----------------------------------
tensor = load_adjacency_tensor(FILE_PT)
base_adj = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
base_graph = nx.from_numpy_array(base_adj)

# ---- static observation & generative-model parameters ------
observations = {'norm': 1, 'MB': 0, 'net': 0, 'tok': 0}
D = {'norm': np.array([0.5, 0.5]),
     'MB'  : np.array([0.5, 0.5]),
     'net' : np.ones(base_graph.number_of_nodes()) /
              base_graph.number_of_nodes(),
     'tok' : np.array([0.5, 0.5])}
A = {'norm': np.array([[0.9, 0.1], [0.1, 0.9]]),
     'MB'  : np.array([[0.8, 0.2], [0.2, 0.8]]),
     'net' : np.eye(base_graph.number_of_nodes()),
     'tok' : np.array([[0.85, 0.15], [0.15, 0.85]])}

# ------------------------------------------------------------
#  (3)  Monte-Carlo loop
# ------------------------------------------------------------
runs = len(R_VALS)
steps = N_STEPS + 1       # include step 0
times_all = np.zeros((N_RUNS, steps, runs))
toks_all  = np.zeros((N_RUNS, steps, runs))

for run in range(N_RUNS):
    np.random.seed(run)
    random.seed(run)

    # fresh copy of initial graph
    G0 = base_graph.copy()

    # optimise with Active Inference
    nets, freeEs = active_inference_optimization(
        N_STEPS, G0, observations, D, A,
        n_candidates=N_CAND, max_additions=MAX_ADD
    )

    # contagion metrics for every stored graph
    for step, net in enumerate(nets):
        for j, r in enumerate(R_VALS):
            t, tk = spread_disease(net, start_node=0, r=r,
                                   return_tokens=True, delay=1)
            times_all[run, step, j] = t
            toks_all [run, step, j] = tk

print("Monte-Carlo collection completed.")

# ------------------------------------------------------------
#  (4)  Helper for shaded plots
# ------------------------------------------------------------
def plot_with_band(ax, x, y_runs, colour, label, alpha=0.18):
    y_mean = y_runs.mean(axis=0)
    y_min  = y_runs.min(axis=0)
    y_max  = y_runs.max(axis=0)
    ax.plot(x, y_mean, color=colour, label=label, lw=2)
    ax.fill_between(x, y_min, y_max, color=colour, alpha=alpha)

cols  = ['blue', 'red', 'green']
labels = [r'$r=0.5$', r'$r=0.75$', r'$r=1.0$']
xvals  = np.arange(steps)

# ------------------------------------------------------------
#  (5)  Contagion-time panel
# ------------------------------------------------------------
fig_ct, ax_ct = plt.subplots(figsize=(8, 3.8))
for j in range(runs):
    plot_with_band(ax_ct, xvals, times_all[:, :, j], cols[j], labels[j])
ax_ct.set_xlabel('Number of Network Reconfiguration Steps')
ax_ct.set_ylabel('Social Contagion of All Nodes (in Time)')
ax_ct.set_title('Contagion Time vs. Network Reconfiguration Steps')
ax_ct.set_xlim(0, N_STEPS)
ax_ct.set_ylim(0, 20)
ax_ct.grid(alpha=.3)
ax_ct.legend()
fig_ct.tight_layout()

# ------------------------------------------------------------
#  (6)  Token-count panel
# ------------------------------------------------------------
fig_tok, ax_tok = plt.subplots(figsize=(8, 3.8))
for j in range(runs):
    plot_with_band(ax_tok, xvals, toks_all[:, :, j], cols[j], labels[j])
ax_tok.set_xlabel('Number of Network Reconfiguration Steps')
ax_tok.set_ylabel('Teleological Token Actuation (in Tokens)')
ax_tok.set_title('Actuated Tokens vs. Network Reconfiguration Steps')
ax_tok.set_xlim(0, N_STEPS)
ax_tok.set_ylim(0, 400)
ax_tok.grid(alpha=.3)
ax_tok.legend()
fig_tok.tight_layout()

plt.show()
