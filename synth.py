"""
synth.py
========
Synthetic benchmark functions from:
  "Neural Interaction Detection" — Tsang et al., ICLR 2018
  https://openreview.net/pdf?id=ByOfBggRZ

Recreates all 6 functions (F1–F6) from Table 1 of the paper.
Each function returns:
    X            : np.ndarray [n_samples x 10], features uniform in [0,1]
    Y            : np.ndarray [n_samples],       target values
    ground_truth : dict with keys
                     'pairwise'  — list of true pairwise interaction tuples (0-indexed)
                     'any_order' — list of true any-order interaction tuples (0-indexed)

Usage:
    import synth
    X, Y, ground_truth = synth.functions[3](num_samples=30000, seed=42, noise_std=0.1)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def _base_X(num_samples: int, num_features: int = 10,
             seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, size=(num_samples, num_features)).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# F1  —  Linear  (no interactions, baseline)
# y = sum(xi)  for i in 1..10
# ─────────────────────────────────────────────────────────────
def f1(num_samples: int, seed: int, noise_std: float):
    rng = np.random.RandomState(seed)
    X = _base_X(num_samples, seed=seed)
    Y = X.sum(axis=1)
    Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)
    ground_truth = {
        "pairwise":   [],
        "any_order":  [(i,) for i in range(10)],
    }
    return X, Y, ground_truth


# ─────────────────────────────────────────────────────────────
# F2  —  Pairwise interactions (simple products)
# y = x1*x2 + x3*x4 + x5*x6 + x7*x8 + x9*x10
# ─────────────────────────────────────────────────────────────
def f2(num_samples: int, seed: int, noise_std: float):
    rng = np.random.RandomState(seed)
    X = _base_X(num_samples, seed=seed)
    Y = (X[:, 0] * X[:, 1] +
         X[:, 2] * X[:, 3] +
         X[:, 4] * X[:, 5] +
         X[:, 6] * X[:, 7] +
         X[:, 8] * X[:, 9])
    Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)
    ground_truth = {
        "pairwise":   [(0,1),(2,3),(4,5),(6,7),(8,9)],
        "any_order":  [(0,1),(2,3),(4,5),(6,7),(8,9)],
    }
    return X, Y, ground_truth


# ─────────────────────────────────────────────────────────────
# F3  —  Mixed interactions  ← THIS IS synth.functions[3]
#         used in Selim's demo
#
# y = π^(x1*x2) * √(2*x3) - arcsin(x4) + log(x3 + x5)
#     - x9 / (1 + x10^2) + sin((x6 - 1/2)*x7*x8)
#     + x7*x8  (implicit higher-order)
#
# True interactions (1-indexed in paper → 0-indexed here):
#   Pairwise  : (0,1), (2,4), (5,6), (5,7), (6,7)
#   Any-order : (0,1), (2,4), (5,6,7)
# ─────────────────────────────────────────────────────────────
def f3(num_samples: int, seed: int, noise_std: float):
    rng = np.random.RandomState(seed)
    X = _base_X(num_samples, seed=seed)
    x1, x2, x3, x4, x5 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]
    x6, x7, x8, x9, x10 = X[:,5], X[:,6], X[:,7], X[:,8], X[:,9]

    Y = (np.pi ** (x1 * x2) * np.sqrt(2 * x3)
         - np.arcsin(x4)
         + np.log(x3 + x5 + 1e-8)
         - (x9 / (1 + x10 ** 2))
         + np.sin((x6 - 0.5) * x7 * x8)
         + x7 * x8)
    Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)

    ground_truth = {
        "pairwise":   [(0,1), (2,4), (5,6), (5,7), (6,7)],
        "any_order":  [(0,1), (2,4), (5,6,7)],
    }
    return X, Y.astype(np.float32), ground_truth


# ─────────────────────────────────────────────────────────────
# F4  —  Higher-order interactions
# y = exp(x1*x2*x3) + exp(x4*x5*x6) + x7 + x8 + x9 + x10
# ─────────────────────────────────────────────────────────────
def f4(num_samples: int, seed: int, noise_std: float):
    rng = np.random.RandomState(seed)
    X = _base_X(num_samples, seed=seed)
    Y = (np.exp(X[:,0] * X[:,1] * X[:,2]) +
         np.exp(X[:,3] * X[:,4] * X[:,5]) +
         X[:,6] + X[:,7] + X[:,8] + X[:,9])
    Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)
    ground_truth = {
        "pairwise":   [(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)],
        "any_order":  [(0,1,2), (3,4,5)],
    }
    return X, Y.astype(np.float32), ground_truth


# ─────────────────────────────────────────────────────────────
# F5  —  Cosine interactions
# y = cos(π*x1*x2) + cos(π*x3*x4*x5) + x6 + x7 + x8 + x9 + x10
# ─────────────────────────────────────────────────────────────
def f5(num_samples: int, seed: int, noise_std: float):
    rng = np.random.RandomState(seed)
    X = _base_X(num_samples, seed=seed)
    Y = (np.cos(np.pi * X[:,0] * X[:,1]) +
         np.cos(np.pi * X[:,2] * X[:,3] * X[:,4]) +
         X[:,5] + X[:,6] + X[:,7] + X[:,8] + X[:,9])
    Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)
    ground_truth = {
        "pairwise":   [(0,1),(2,3),(2,4),(3,4)],
        "any_order":  [(0,1),(2,3,4)],
    }
    return X, Y.astype(np.float32), ground_truth


# ─────────────────────────────────────────────────────────────
# F6  —  All pairwise (dense)
# y = sum of x_i * x_j  for all i < j  (first 5 features)
# ─────────────────────────────────────────────────────────────
def f6(num_samples: int, seed: int, noise_std: float):
    rng = np.random.RandomState(seed)
    X = _base_X(num_samples, seed=seed)
    Y = np.zeros(num_samples, dtype=np.float32)
    pairs = []
    for i in range(5):
        for j in range(i+1, 5):
            Y += X[:, i] * X[:, j]
            pairs.append((i, j))
    Y = Y + rng.normal(0, noise_std, size=Y.shape).astype(np.float32)
    ground_truth = {
        "pairwise":  pairs,
        "any_order": pairs,
    }
    return X, Y, ground_truth


# ─────────────────────────────────────────────────────────────
# Registry — matches synth.functions[N] indexing from demo
# Index: 0=f1, 1=f2, 2=f3, 3=f3(★), 4=f4, 5=f5, 6=f6
# Note: demo uses functions[3] which is F3
# ─────────────────────────────────────────────────────────────
functions = [f1, f2, f3, f3, f4, f5, f6]


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing synth.functions[3] (F3) — as used in Selim's demo\n")
    X, Y, gt = functions[3](num_samples=30000, seed=42, noise_std=0.1)
    print(f"X shape : {X.shape}")
    print(f"Y shape : {Y.shape}")
    print(f"Y mean  : {Y.mean():.4f}  std: {Y.std():.4f}")
    print(f"Pairwise interactions : {gt['pairwise']}")
    print(f"Any-order interactions: {gt['any_order']}")
    print("\nAll functions OK:")
    for i, fn in enumerate(functions):
        X_, Y_, _ = fn(num_samples=1000, seed=0, noise_std=0.1)
        print(f"  functions[{i}] ({fn.__name__}): X{X_.shape} Y{Y_.shape}")
