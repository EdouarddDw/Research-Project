"""
pdp_analysis.py
======================

Partial Dependence Plot (PDP) analysis for MLP snapshots.

Loads saved model snapshots from:
    outputs/snapshots/unregularized/model_epoch_{N}.pt

Reuses the synthetic data generation and MLP architecture from
`train_mlp_snapshots.py` to compute Monte Carlo PDPs that show how
feature interactions evolve during training.

Generated figures (saved under ./outputs/snapshots/):
    - pdp_1d_evolution.png
    - pdp_2d_modifier.png
    - pdp_2d_crossover.png
    - pdp_modifier_vs_crossover.png
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_mlp_snapshots import generate_data, MLP, N_FEATURES, SEED


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_OUTPUT_DIR = "./outputs/snapshots"
UNREG_SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "unregularized")

# Epochs for which we expect model_epoch_{epoch}.pt to exist
SNAPSHOT_EPOCHS_1D = [1, 10, 50, 100, 300]
SNAPSHOT_EPOCHS_2D = [1, 50, 300]
INTERACTION_EPOCH = 300

GRID_POINTS_1D = 50
GRID_POINTS_2D = 40
MC_MAX_SAMPLES = 1000  # number of background samples for Monte Carlo PDP


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def ensure_dirs():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    if not os.path.isdir(UNREG_SNAPSHOT_DIR):
        raise FileNotFoundError(
            f"Snapshot directory not found: {UNREG_SNAPSHOT_DIR}. "
            "Run train_mlp_snapshots.py first to generate model snapshots."
        )


def load_unreg_models(epochs: list[int]) -> dict[int, MLP]:
    """
    Load unregularized models for the given snapshot epochs.
    """
    models: dict[int, MLP] = {}
    for epoch in epochs:
        model_path = os.path.join(UNREG_SNAPSHOT_DIR, f"model_epoch_{epoch}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing snapshot: {model_path}. "
                "Make sure train_mlp_snapshots.py has been run with the "
                "unregularized configuration."
            )

        model = MLP(N_FEATURES)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models[epoch] = model
        print(f"Loaded model snapshot for epoch {epoch} from {model_path}")

    return models


def get_background_data(mc_max_samples: int = MC_MAX_SAMPLES) -> tuple[np.ndarray, list[str]]:
    """
    Regenerate the same synthetic dataset used for training and
    return a (possibly subsampled) background matrix X for Monte Carlo
    integration, along with feature names.
    """
    data = generate_data(n_samples=3000, n_true=5, n_noise=5, seed=SEED)
    X = data["X"]  # shape [n_samples, n_features]
    feature_names = data["feature_names"]

    n_samples = X.shape[0]
    if n_samples > mc_max_samples:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n_samples, size=mc_max_samples, replace=False)
        X_bg = X[idx]
    else:
        X_bg = X

    return X_bg, feature_names


def model_predict(model: MLP, X: np.ndarray) -> np.ndarray:
    """
    Run the PyTorch model on a NumPy array and return predictions as NumPy.
    """
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32))
        preds = model(x_t).cpu().numpy()
    return preds


def compute_1d_pdp(model: MLP,
                   X_bg: np.ndarray,
                   feature_idx: int,
                   grid: np.ndarray) -> np.ndarray:
    """
    Monte Carlo PDP for a single feature:
    For each grid value v, set X[:, feature_idx]=v and average predictions.
    """
    X_mod = X_bg.copy()
    pdp_vals = []
    for v in grid:
        X_mod[:, feature_idx] = v
        preds = model_predict(model, X_mod)
        pdp_vals.append(preds.mean())
    return np.array(pdp_vals)


def compute_2d_pdp(model: MLP,
                   X_bg: np.ndarray,
                   i: int,
                   j: int,
                   grid_points: int = GRID_POINTS_2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo 2D PDP for a pair of features (i, j).
    For each (v_i, v_j) on a grid, set X[:, i]=v_i, X[:, j]=v_j and
    average predictions.
    """
    xs = np.linspace(0.0, 1.0, grid_points)
    ys = np.linspace(0.0, 1.0, grid_points)
    X_mod = X_bg.copy()
    Z = np.zeros((grid_points, grid_points), dtype=np.float32)

    for ix, xv in enumerate(xs):
        for iy, yv in enumerate(ys):
            X_mod[:, i] = xv
            X_mod[:, j] = yv
            preds = model_predict(model, X_mod)
            Z[iy, ix] = preds.mean()

    return xs, ys, Z


def compute_interaction_curves(model: MLP,
                               X_bg: np.ndarray,
                               var_idx: int,
                               cond_idx: int,
                               cond_values: list[float],
                               grid_points: int = GRID_POINTS_1D) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """
    For a variable var_idx and conditioning variable cond_idx, compute
    PDP curves of var_idx at several fixed values of cond_idx.
    """
    grid = np.linspace(0.0, 1.0, grid_points)
    X_mod = X_bg.copy()
    curves: dict[float, np.ndarray] = {}

    for c in cond_values:
        values = []
        for v in grid:
            X_mod[:, var_idx] = v
            X_mod[:, cond_idx] = c
            preds = model_predict(model, X_mod)
            values.append(preds.mean())
        curves[c] = np.array(values)

    return grid, curves


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------
def plot_1d_pdp_evolution(models: dict[int, MLP],
                          X_bg: np.ndarray,
                          feature_names: list[str]):
    """
    1D PDP plots for the 5 true features across multiple epochs.
    All epochs are plotted as overlapping lines on the same axes
    (one subplot per feature).
    """
    epochs = sorted(models.keys())
    grid = np.linspace(0.0, 1.0, GRID_POINTS_1D)

    n_features_to_plot = 5  # first 5 = true features
    fig, axes = plt.subplots(
        1, n_features_to_plot, figsize=(4 * n_features_to_plot, 4), sharey=True
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(epochs)))

    for feat_idx in range(n_features_to_plot):
        ax = axes[feat_idx]
        for color, epoch in zip(colors, epochs):
            model = models[epoch]
            pdp_vals = compute_1d_pdp(model, X_bg, feat_idx, grid)
            ax.plot(grid, pdp_vals, label=f"Epoch {epoch}", color=color)

        ax.set_xlabel(feature_names[feat_idx])
        if feat_idx == 0:
            ax.set_ylabel("Model output (PDP)")
        ax.set_title(f"Feature: {feature_names[feat_idx]}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(epochs))
    fig.suptitle("1D PDP Evolution: True Features", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.90])

    out_path = os.path.join(BASE_OUTPUT_DIR, "pdp_1d_evolution.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_2d_pdp_modifier(models: dict[int, MLP],
                         X_bg: np.ndarray):
    """
    2D PDP heatmaps for the modifier interaction (x0, x1)
    at epochs [1, 50, 300].
    """
    epochs = SNAPSHOT_EPOCHS_2D
    fig, axes = plt.subplots(1, len(epochs), figsize=(5 * len(epochs), 4))

    for ax, epoch in zip(axes, epochs):
        model = models[epoch]
        xs, ys, Z = compute_2d_pdp(model, X_bg, i=0, j=1)
        Xg, Yg = np.meshgrid(xs, ys)
        cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="viridis")
        ax.set_xlabel("x0 (modifier)")
        ax.set_ylabel("x1")
        ax.set_title(f"Epoch {epoch}")
        fig.colorbar(cf, ax=ax)

    fig.suptitle("Modifier Interaction PDP (x0, x1)", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])

    out_path = os.path.join(BASE_OUTPUT_DIR, "pdp_2d_modifier.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_2d_pdp_crossover(models: dict[int, MLP],
                          X_bg: np.ndarray):
    """
    2D PDP heatmaps for the crossover interaction (x2, x3)
    at epochs [1, 50, 300].
    """
    epochs = SNAPSHOT_EPOCHS_2D
    fig, axes = plt.subplots(1, len(epochs), figsize=(5 * len(epochs), 4))

    for ax, epoch in zip(axes, epochs):
        model = models[epoch]
        xs, ys, Z = compute_2d_pdp(model, X_bg, i=2, j=3)
        Xg, Yg = np.meshgrid(xs, ys)
        cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="coolwarm")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("x2")
        ax.set_ylabel("x3 (crossover)")
        ax.set_title(f"Epoch {epoch}")
        fig.colorbar(cf, ax=ax)

    fig.suptitle("Crossover Interaction PDP (x2, x3)", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])

    out_path = os.path.join(BASE_OUTPUT_DIR, "pdp_2d_crossover.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_modifier_vs_crossover_signature(model: MLP,
                                         X_bg: np.ndarray):
    """
    PDP interaction curves at a single epoch (typically 300) to
    highlight modifier vs crossover signatures.

    Modifier (x0, x1): lines for PDP(x0) at x1 in {0.1, 0.5, 0.9}
    should fan out but remain with consistent sign.

    Crossover (x2, x3): lines for PDP(x2) at x3 in {0.1, 0.5, 0.9}
    should cross each other (sign reversal).
    """
    cond_values = [0.1, 0.5, 0.9]

    # Modifier: x0 varying, x1 conditioned
    grid_x0, modifier_curves = compute_interaction_curves(
        model, X_bg, var_idx=0, cond_idx=1, cond_values=cond_values
    )

    # Crossover: x2 varying, x3 conditioned
    grid_x2, crossover_curves = compute_interaction_curves(
        model, X_bg, var_idx=2, cond_idx=3, cond_values=cond_values
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Left: Modifier
    ax_mod = axes[0]
    for c in cond_values:
        ax_mod.plot(grid_x0, modifier_curves[c], label=f"x1 = {c:.1f}")
    ax_mod.set_xlabel("x0")
    ax_mod.set_ylabel("Model output (PDP)")
    ax_mod.set_title("Modifier Interaction (x0 | x1)")
    ax_mod.legend()

    # Right: Crossover
    ax_cross = axes[1]
    for c in cond_values:
        ax_cross.plot(grid_x2, crossover_curves[c], label=f"x3 = {c:.1f}")
    ax_cross.set_xlabel("x2")
    ax_cross.set_title("Crossover Interaction (x2 | x3)")
    ax_cross.legend()

    fig.suptitle("Modifier vs Crossover Interaction Signature", fontsize=14)
    fig.tight_layout(rect=[0, 0.0, 1, 0.92])

    out_path = os.path.join(BASE_OUTPUT_DIR, "pdp_modifier_vs_crossover.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PDP Analysis for MLP Snapshots")
    print("=" * 60)

    ensure_dirs()

    # Load data and background samples for Monte Carlo PDP
    X_bg, feature_names = get_background_data(MC_MAX_SAMPLES)
    print(f"Background data shape for PDP: {X_bg.shape}")

    # Load models for all required epochs (union of 1D/2D/interaction)
    epochs_needed = sorted(
        set(SNAPSHOT_EPOCHS_1D) | set(SNAPSHOT_EPOCHS_2D) | {INTERACTION_EPOCH}
    )
    models = load_unreg_models(epochs_needed)

    # 1D PDP evolution for true features
    print("\nComputing 1D PDP evolution...")
    plot_1d_pdp_evolution(
        {e: models[e] for e in SNAPSHOT_EPOCHS_1D}, X_bg, feature_names
    )

    # 2D PDPs for modifier and crossover interactions
    print("\nComputing 2D PDP for modifier interaction (x0, x1)...")
    plot_2d_pdp_modifier(models, X_bg)

    print("\nComputing 2D PDP for crossover interaction (x2, x3)...")
    plot_2d_pdp_crossover(models, X_bg)

    # Modifier vs crossover interaction signature at final epoch
    print(f"\nComputing interaction signature at epoch {INTERACTION_EPOCH}...")
    model_final = models[INTERACTION_EPOCH]
    plot_modifier_vs_crossover_signature(model_final, X_bg)

    print("\nPDP analysis complete. Figures saved under:", BASE_OUTPUT_DIR)


if __name__ == "__main__":
    main()

