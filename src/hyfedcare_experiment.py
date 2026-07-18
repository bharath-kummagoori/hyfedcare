"""
HyFedCare experiments
=====================

Companion code for the book chapter:
"A Hybrid Federated Learning Framework With Tier-Adaptive Privacy-Preserving
Techniques: Secure IoT-Enabled Smart Healthcare"

This script runs three experiment arms on the Wisconsin Diagnostic Breast
Cancer (WDBC) dataset with simulated non-IID hospital clients:

  Arm 1  Client-level DP-FedAvg
         Each client's whole update is clipped, so the resulting (eps, delta)
         bound protects an entire client's dataset (institution-level).

  Arm 2  Record-level federated DP-SGD
         Every per-example gradient is clipped before aggregation, so the
         same accountant yields a genuine per-patient-record guarantee.

  Arm 3  Noise placement comparison
         Same noise multiplier applied once at the coordination tier vs
         independently at every device, quantifying the utility price of
         pushing noise down to the device tier.

Run from the repository root:

    python3 src/hyfedcare_experiment.py

Results are written to results/experiment_results.json. Every run is fully
seeded, so the numbers reproduce exactly.
"""

import json
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)   # master seed for the data partition

K = 20          # number of simulated hospital clients
EDGES = 4       # edge gateways used in the hierarchical communication model
T = 40          # federated training rounds
E_LOCAL = 2     # local epochs per round (Arm 1 only)
LR = 0.5        # learning rate for local training
CLIP = 1.0      # L2 clipping bound for a whole client update (Arm 1)
DELTA = 1e-5    # DP delta, fixed well below 1/n
SIGMAS = [0.0, 1.0, 2.0, 4.0, 8.0]   # Gaussian noise multipliers to sweep
SEEDS = [0, 1, 2, 3, 4]              # per-configuration repetitions

RESULTS_PATH = os.path.join('results', 'experiment_results.json')


# ---------------------------------------------------------------------------
# Data loading and non-IID partitioning
# ---------------------------------------------------------------------------
def load_data():
    """Load WDBC, make a stratified 75/25 train/test split and standardize.

    WDBC ships with scikit-learn, so no download step is needed. Features are
    z-scored with statistics fitted on the training split only, to avoid any
    leakage into the test set.
    """
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


def dirichlet_partition(y, k, alpha=0.5, rng=RNG):
    """Split training indices across k clients with Dirichlet label skew.

    Smaller alpha produces stronger heterogeneity. At alpha = 0.5 some
    clients end up with only a handful of samples and some lack one class
    entirely - deliberately so, because that is what real multi-hospital
    cohorts look like and it is the regime where federation matters.

    Returns a list of index arrays, one per client.
    """
    idx_by_class = [np.where(y == c)[0] for c in np.unique(y)]
    client_idx = [[] for _ in range(k)]
    for idx_c in idx_by_class:
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * k)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx_c, cuts)):
            client_idx[cid].extend(part.tolist())
    return [np.array(ci) for ci in client_idx]


# ---------------------------------------------------------------------------
# Model: logistic regression trained with plain gradient descent
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Numerically clipped logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def add_bias(A):
    """Append a constant-1 column so the bias is just one more weight."""
    return np.hstack([A, np.ones((A.shape[0], 1))])


def grad(w, A, t):
    """Mean logistic-loss gradient over a batch (A, t)."""
    return A.T @ (sigmoid(A @ w) - t) / len(t)


def local_train(w0, A, t, epochs=E_LOCAL, lr=LR):
    """Run full-batch gradient descent locally and return the new weights.

    Used both for client-side training in FedAvg (few epochs from the
    current global model) and, with a large epoch count, for the centralized
    and local-only baselines.
    """
    w = w0.copy()
    for _ in range(epochs):
        w -= lr * grad(w, A, t)
    return w


def evaluate(w, A, t):
    """Return (accuracy, ROC-AUC) of weights w on the test set."""
    p = sigmoid(A @ w)
    return accuracy_score(t, p > 0.5), roc_auc_score(t, p)


def sample_losses(w, A, t):
    """Per-sample cross-entropy losses - the signal the MIA thresholds on."""
    p = np.clip(sigmoid(A @ w), 1e-9, 1 - 1e-9)
    return -(t * np.log(p) + (1 - t) * np.log(1 - p))


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------
def epsilon_zcdp(sigma, rounds, delta=DELTA):
    """Convert a per-round Gaussian mechanism into a total (eps, delta).

    With full participation, each round releases one Gaussian-noised sum
    whose sensitivity equals the clipping bound. In zero-concentrated DP
    that is rho = 1 / (2 sigma^2) per round; rho composes additively over
    rounds, and the standard conversion gives
        eps = rho + 2 * sqrt(rho * ln(1/delta)).
    The granularity of the guarantee (client vs record) is decided by WHAT
    was clipped, not by this function - see the two run_* functions below.
    """
    if sigma == 0:
        return float('inf')
    rho = rounds / (2.0 * sigma ** 2)
    return rho + 2.0 * np.sqrt(rho * np.log(1.0 / delta))


def mia_auc(w, A_members, y_members, A_non, y_non):
    """Loss-threshold membership inference attack.

    Scores every sample by negative loss (members tend to have lower loss)
    and reports the AUC of separating training members from test
    non-members. This is the weakest credible attack; it serves as a floor
    measurement, not as proof of protection.
    """
    lo_m = sample_losses(w, A_members, y_members)
    lo_n = sample_losses(w, A_non, y_non)
    scores = np.concatenate([-lo_m, -lo_n])
    labels = np.concatenate([np.ones(len(lo_m)), np.zeros(len(lo_n))])
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# Arm 1 / Arm 3: client-level DP-FedAvg
# ---------------------------------------------------------------------------
def run_fedavg_client_dp(client_data, d, sigma, seed=0, placement='central'):
    """DP-FedAvg with whole-update clipping (client-level guarantee).

    Each round every client trains locally for E_LOCAL epochs, its update
    delta_w is clipped to L2 norm CLIP, and Gaussian noise is added either
    once to the summed update ('central', the coordination-tier default) or
    independently to every client's update ('local', the device-tier
    variant used by Arm 3). The server then averages.

    Because the clipped object is a whole client update, the accountant's
    (eps, delta) protects the client's entire dataset - NOT any individual
    patient record. The chapter is explicit about this distinction.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(d + 1)
    for _ in range(T):
        updates = []
        for A_c, t_c in client_data:
            delta_w = local_train(w, A_c, t_c) - w
            nrm = np.linalg.norm(delta_w)
            if nrm > CLIP:
                delta_w *= CLIP / nrm
            if placement == 'local' and sigma > 0:
                delta_w = delta_w + rng.normal(0, sigma * CLIP, delta_w.shape)
            updates.append(delta_w)
        agg = np.sum(updates, axis=0)
        if placement == 'central' and sigma > 0:
            agg = agg + rng.normal(0, sigma * CLIP, agg.shape)
        w = w + agg / K
    return w


# ---------------------------------------------------------------------------
# Arm 2: record-level federated DP-SGD
# ---------------------------------------------------------------------------
def calibrate_record_clip(client_data, d):
    """Pick the per-example clipping bound C_rec by the pilot rule.

    Following the parameter-selection discipline described in the chapter,
    C_rec is set to the median per-example gradient norm measured at the
    initial model (w = 0) across all clients. Median rather than max keeps
    the bound tight without destroying most examples' signal.
    """
    norms = []
    w0 = np.zeros(d + 1)
    for A_c, t_c in client_data:
        g = A_c * (sigmoid(A_c @ w0) - t_c)[:, None]
        norms.extend(np.linalg.norm(g, axis=1).tolist())
    return float(np.median(norms))


def run_fedsgd_record_dp(client_data, d, n_total, c_rec, sigma, seed=0,
                         lr=0.5):
    """Federated DP-SGD with per-example clipping (record-level guarantee).

    Each round every client computes per-example gradients at the current
    global model, clips EACH example's gradient to c_rec, and sends the
    clipped sum. The coordination tier adds one Gaussian noise draw to the
    global sum and takes a gradient step.

    Adding or removing one patient record changes the global sum by at most
    c_rec, so the accountant's (eps, delta) is a true per-record bound -
    the granularity patient-facing privacy promises actually require.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(d + 1)
    for _ in range(T):
        total = np.zeros(d + 1)
        for A_c, t_c in client_data:
            g = A_c * (sigmoid(A_c @ w) - t_c)[:, None]     # per-example grads
            nrms = np.linalg.norm(g, axis=1, keepdims=True)
            g = g * np.minimum(1.0, c_rec / np.maximum(nrms, 1e-12))
            total += g.sum(axis=0)
        if sigma > 0:
            total = total + rng.normal(0, sigma * c_rec, total.shape)
        w = w - lr * total / n_total
    return w


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def sweep(runner, A_tr, y_tr, A_te, y_te, **kw):
    """Run a configuration across SIGMAS x SEEDS and aggregate the metrics."""
    rows = []
    for s in SIGMAS:
        accs, aucs, mias = [], [], []
        for seed in SEEDS:
            w = runner(sigma=s, seed=seed, **kw)
            a, u = evaluate(w, A_te, y_te)
            accs.append(a)
            aucs.append(u)
            mias.append(mia_auc(w, A_tr, y_tr, A_te, y_te))
        rows.append({
            'sigma': s, 'epsilon': epsilon_zcdp(s, T),
            'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
            'mia_mean': float(np.mean(mias)), 'mia_std': float(np.std(mias)),
        })
    return rows


def main():
    X_tr, X_te, y_tr, y_te = load_data()
    n, d = X_tr.shape
    clients = dirichlet_partition(y_tr, K)
    A_tr, A_te = add_bias(X_tr), add_bias(X_te)
    client_data = [(add_bias(X_tr[ci]), y_tr[ci]) for ci in clients]

    # Baselines: the centralized oracle (data pooling, unavailable in
    # practice) and isolated local-only training at each viable client.
    w_central = local_train(np.zeros(d + 1), A_tr, y_tr, epochs=T * E_LOCAL)
    acc_c, auc_c = evaluate(w_central, A_te, y_te)

    local_accs, local_aucs = [], []
    for A_c, t_c in client_data:
        if len(np.unique(t_c)) < 2 or len(t_c) < 5:
            continue    # a client missing a class cannot train alone at all
        w_l = local_train(np.zeros(d + 1), A_c, t_c, epochs=T * E_LOCAL)
        a, u = evaluate(w_l, A_te, y_te)
        local_accs.append(a)
        local_aucs.append(u)

    # Arm 1: client-level DP-FedAvg with coordination-tier noise.
    arm1 = sweep(
        lambda sigma, seed: run_fedavg_client_dp(client_data, d, sigma, seed),
        A_tr, y_tr, A_te, y_te)

    # Arm 2: record-level DP-SGD with per-example clipping.
    c_rec = calibrate_record_clip(client_data, d)
    arm2 = sweep(
        lambda sigma, seed: run_fedsgd_record_dp(
            client_data, d, n, c_rec, sigma, seed),
        A_tr, y_tr, A_te, y_te)

    # Arm 3: same multiplier, noise moved from coordination tier to devices.
    placement = []
    for pl in ['central', 'local']:
        accs = [evaluate(run_fedavg_client_dp(client_data, d, 4.0, seed, pl),
                         A_te, y_te)[0] for seed in SEEDS]
        placement.append({
            'placement': pl, 'sigma': 4.0,
            'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs))})

    # Communication accounting for flat vs hierarchical aggregation. The
    # K/E ratio is topology arithmetic, independent of model size.
    model_bytes = (d + 1) * 4
    comm = {
        'model_bytes': model_bytes, 'rounds': T,
        'flat_wan_msgs_per_round': 2 * K,
        'hier_wan_msgs_per_round': 2 * EDGES,
        'hier_lan_msgs_per_round': 2 * K,
        'flat_wan_bytes_total': 2 * K * T * model_bytes,
        'hier_wan_bytes_total': 2 * EDGES * T * model_bytes,
    }

    out = {
        'dataset': 'WDBC (569 samples, 30 features)',
        'n_train': int(n), 'n_test': int(len(y_te)),
        'clients': K, 'edges': EDGES, 'rounds': T,
        'clip_client': CLIP, 'clip_record': c_rec, 'delta': DELTA,
        'client_sizes': [len(c) for c in clients],
        'centralized': {'acc': float(acc_c), 'auc': float(auc_c)},
        'local_only': {
            'acc_mean': float(np.mean(local_accs)),
            'acc_std': float(np.std(local_accs)),
            'auc_mean': float(np.mean(local_aucs)),
            'n_viable': len(local_accs)},
        'client_level_central': arm1,
        'record_level': arm2,
        'noise_placement': placement,
        'communication': comm,
    }

    os.makedirs('results', exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Written {RESULTS_PATH}')
    print(f"Record-level DP @ eps~4.1: acc = {arm2[-1]['acc_mean']:.4f}")
    print(f"Client-level DP @ eps~4.1: acc = {arm1[-1]['acc_mean']:.4f}")


if __name__ == '__main__':
    main()
