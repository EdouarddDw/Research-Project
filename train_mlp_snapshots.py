"""
train_mlp_snapshots.py
======================
Group Project: The Evolution of Logic in AI
-------------------------------------------
This module handles:
  1. Synthetic data generation with known ground-truth interactions
  2. MLP training (PyTorch) with epoch-level "snapshots"
  3. Saving snapshots

Usage:
    python train_mlp_snapshots.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use("Agg")

from plots import plot_loss_curves, find_overfitting_epoch, plot_comparison_loss_curves

# ─────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N_SAMPLES    = 3000        # total samples
N_TRUE       = 5           # informative features
N_NOISE      = 5           # pure noise features  → indices [5..9]
N_FEATURES   = N_TRUE + N_NOISE

EPOCHS       = 300         # train long enough to reach "Decay" phase
LR           = 1e-3
BATCH_SIZE   = 256
VAL_SPLIT    = 0.2

# Snapshot epochs (logarithmic spacing + key milestones)
SNAPSHOT_EPOCHS = sorted(set(
    [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]
))

# Base output directory; individual runs write to subfolders under this
BASE_OUTPUT_DIR = "./outputs/snapshots"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
def generate_data(n_samples: int, n_true: int, n_noise: int,
                  noise_std: float = 0.3, seed: int = 42) -> dict:
    """
    Generate synthetic dataset with known ground-truth interactions.

    Signal features (x0..x4):
      - Modifier interaction  : x0 * x1          (x1 modulates strength of x0)
      - Crossover interaction : x2 * (x3 - 0.5)  (sign of x2 effect flips around x3=0.5)
      - Simple nonlinear      : sin(2*pi*x4)

    Noise features (x5..x9): pure Gaussian noise, zero true effect.

    Ground truth interactions (for AUC / R-Precision benchmarking):
      Pairwise: (0,1), (2,3)
      Any-order: (0,1), (2,3), (4,)
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_samples, n_true + n_noise)).astype(np.float32)

    # ── signal ──
    modifier_term  = X[:, 0] * X[:, 1]                      # Modifier
    crossover_term = X[:, 2] * (X[:, 3] - 0.5)              # Crossover
    nonlinear_term = np.sin(2 * np.pi * X[:, 4])            # Nonlinear main effect

    y = modifier_term + crossover_term + nonlinear_term
    y += rng.normal(0, noise_std, size=n_samples).astype(np.float32)
    y = y.astype(np.float32)

    ground_truth = {
        "pairwise":   [(0, 1), (2, 3)],
        "any_order":  [(0, 1), (2, 3), (4,)],
        "true_idx":   list(range(n_true)),
        "noise_idx":  list(range(n_true, n_true + n_noise)),
    }

    feature_names = (
        [f"x{i}_true"  for i in range(n_true)] +
        [f"x{i}_noise" for i in range(n_noise)]
    )

    return {"X": X, "y": y, "ground_truth": ground_truth,
            "feature_names": feature_names}


# ─────────────────────────────────────────────
# 2.  MLP ARCHITECTURE
# ─────────────────────────────────────────────
class MLP(nn.Module):
    """
    Fully-connected MLP with optional Dropout regularization.
    Architecture mirrors the NID demo: [140, 100, 60, 20].
    """
    def __init__(self, n_features: int, hidden: list = None,
                 dropout_p: float = 0.0):
        super().__init__()
        if hidden is None:
            hidden = [140, 100, 60, 20]

        layers = []
        in_dim = n_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────
# 3.  TRAINING LOOP WITH SNAPSHOTS
# ─────────────────────────────────────────────
def build_loaders(X: np.ndarray, y: np.ndarray,
                  val_split: float = 0.2, batch_size: int = 256):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    dataset = TensorDataset(X_t, y_t)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_with_snapshots(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         feature_names: list,
                         ground_truth: dict,
                         epochs: int = EPOCHS,
                         lr: float = LR,
                         l2_lambda: float = 0.0,
                         l1_lambda: float = 0.0,
                         snapshot_epochs: list = None,
                         output_dir: str = BASE_OUTPUT_DIR) -> dict:
    """
    Train MLP and capture snapshots at specified epochs.

    Returns dict with:
      - loss_curves: DataFrame(epoch, train_loss, val_loss)
    """
    if snapshot_epochs is None:
        snapshot_epochs = SNAPSHOT_EPOCHS

    os.makedirs(output_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=l2_lambda)
    criterion = nn.MSELoss()

    loss_records   = []

    print(f"Training for {epochs} epochs | "
          f"Snapshots at: {snapshot_epochs}")
    print(f"Regularization: L2={l2_lambda}, L1={l1_lambda}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):

        # ── train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)

            # L1 regularization (manual)
            if l1_lambda > 0:
                l1_penalty = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_penalty

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)

        train_loss /= len(train_loader.dataset)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        loss_records.append({"epoch": epoch,
                              "train_loss": train_loss,
                              "val_loss": val_loss})

        if epoch % 50 == 0 or epoch <= 10:
            print(f"[Epoch {epoch:>4}] train: {train_loss:.5f} | "
                  f"val: {val_loss:.5f}")

        # ── snapshot ──
        if epoch in snapshot_epochs:
            print(f"  → Snapshot @ epoch {epoch}")
            # Save model weights
            model_path = os.path.join(output_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            model.eval()

    # ── save outputs ──
    loss_df = pd.DataFrame(loss_records)

    # loss_df.to_csv(os.path.join(output_dir, "loss_curves.csv"), index=False) DO WE NEED THIS?

    # Save feature names + ground truth for downstream use (PDP etc.)
    meta = {"feature_names": feature_names, "ground_truth": ground_truth}
    with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("\nTraining complete. Files saved to:", output_dir)
    return {
        "loss_curves": loss_df
    }

# ─────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Evolution of Logic in AI — MLP Training + Snapshots")
    print("=" * 60)

    # ── Data ──
    data = generate_data(N_SAMPLES, N_TRUE, N_NOISE, seed=SEED)
    X, y = data["X"], data["y"]
    feature_names = data["feature_names"]
    ground_truth  = data["ground_truth"]

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  True features : {ground_truth['true_idx']}")
    print(f"  Noise features: {ground_truth['noise_idx']}")
    print(f"  Interactions  : Modifier=(x0,x1)  Crossover=(x2,x3)\n")

    train_loader, val_loader = build_loaders(X, y, VAL_SPLIT, BATCH_SIZE)

    # Output subdirectories for each run
    unreg_dir = os.path.join(BASE_OUTPUT_DIR, "unregularized")
    l2_dir = os.path.join(BASE_OUTPUT_DIR, "l2")
    os.makedirs(unreg_dir, exist_ok=True)
    os.makedirs(l2_dir, exist_ok=True)

    def run_experiment(name: str,
                       l2_lambda: float,
                       output_dir: str) -> dict:
        print("\n" + "=" * 60)
        print(f"Run: {name} (L2={l2_lambda})")
        print("=" * 60)

        model = MLP(N_FEATURES, hidden=[140, 100, 60, 20], dropout_p=0.0)
        print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")

        results = train_with_snapshots(
            model, train_loader, val_loader,
            feature_names=feature_names,
            ground_truth=ground_truth,
            epochs=EPOCHS,
            lr=LR,
            l2_lambda=l2_lambda,
            l1_lambda=0.0,
            snapshot_epochs=SNAPSHOT_EPOCHS,
            output_dir=output_dir,
        )

        print(f"\nGenerating diagnostic plots for {name}...")
        plot_loss_curves(results["loss_curves"], output_dir=output_dir)
        return results

    # Run 1: Unregularized (L2 = 0.0)
    unreg_results = run_experiment(
        name="Unregularized",
        l2_lambda=0.0,
        output_dir=unreg_dir,
    )

    # Run 2: L2-regularized (L2 = 1e-4)
    l2_results = run_experiment(
        name="L2 Regularized",
        l2_lambda=1e-4,
        output_dir=l2_dir,
    )

    # Comparison plots using shared data / epochs
    print("\nGenerating comparison plots...")
    overfit_epoch = find_overfitting_epoch(unreg_results["loss_curves"])
    plot_comparison_loss_curves(
        unreg_results["loss_curves"],
        l2_results["loss_curves"],
        save_path=os.path.join(BASE_OUTPUT_DIR, "comparison_loss_curves.png"),
    )


if __name__ == "__main__":
    main()
