"""
pdp_analysis.py
========================

Partial Dependence Plot (PDP) analysis for MLP snapshots.
Publication-quality matplotlib styling and static PNG figures.

Run AFTER train_mlp_snapshots.py has generated model snapshots in:
    outputs/snapshots/unregularized/model_epoch_{N}.pt

Generated figures (saved under ./outputs/snapshots/):
    - pdp_1d_evolution.png       — 1D PDP grid (signal features × epochs)
    - pdp_2d_grid.png            — 2D PDP grid (GT pairs × selected epochs)
    - pdp_interaction_signature.png — conditional PDP curves (static)

For GIF animations run separately: python pdp_gifs.py
"""

import os
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from multilayer_perceptron import MLP
from train_mlp_snapshots import (
    generate_data_from_synth,
    SYNTH_FN_IDX,
    NOISE_STD,
    NUM_FEATURES,
    NUM_SAMPLES,
    HIDDEN_UNITS,
    USE_MAIN_EFFECT_NETS,
    SEED,
    SNAPSHOT_EPOCHS,
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE — typography, layout, colours
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          10,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.6,
    "grid.linestyle":     "--",
    "legend.framealpha":  0.85,
    "legend.edgecolor":   "#cccccc",
    "legend.fontsize":    9,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

COND_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # for conditioned curves

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR    = "./outputs/snapshots"
UNREG_SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "unregularized")

# Subset of epochs for static 1D PDP (clear legend) and 2D grid columns
SNAPSHOT_EPOCHS_1D = [1, 10, 30, 75, 150, 300]
GRID_EPOCHS_2D     = [1, 10, 30, 75, 150, 300]   # columns in pdp_2d_grid.png
INTERACTION_EPOCH  = 300

GRID_POINTS_1D = 60
GRID_POINTS_2D = 30   # 30×30 быстрее 40×40, картинка чуть грубее
MC_MAX_SAMPLES = 600  # меньше сэмплов = быстрее 2D PDP и GIF
GIF_DPI = 100
GIF_FPS = 4
GIF_INTERVAL_MS = 250


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    if not os.path.isdir(UNREG_SNAPSHOT_DIR):
        raise FileNotFoundError(
            f"Snapshot directory not found: {UNREG_SNAPSHOT_DIR}\n"
            "Run train_mlp_snapshots.py first."
        )


def load_meta():
    """Load feature_names and ground_truth from meta.pkl saved by train_mlp_snapshots."""
    meta_path = os.path.join(UNREG_SNAPSHOT_DIR, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["feature_names"], meta["ground_truth"]


def load_unreg_models(epochs):
    models = {}
    for epoch in epochs:
        path = os.path.join(UNREG_SNAPSHOT_DIR, f"model_epoch_{epoch}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing snapshot: {path}")
        model = MLP(NUM_FEATURES, HIDDEN_UNITS, use_main_effect_nets=USE_MAIN_EFFECT_NETS)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[epoch] = model
        print(f"  Loaded epoch {epoch}")
    return models


def get_background_data(mc_max_samples=MC_MAX_SAMPLES):
    data = generate_data_from_synth(SYNTH_FN_IDX, NUM_SAMPLES, SEED, NOISE_STD)
    X = data["X"]
    feature_names = data["feature_names"]
    rng = np.random.RandomState(SEED)
    if X.shape[0] > mc_max_samples:
        idx = rng.choice(X.shape[0], mc_max_samples, replace=False)
        X = X[idx]
    return X, feature_names


def model_predict(model, X):
    with torch.no_grad():
        preds = model(torch.from_numpy(X.astype(np.float32)))
        preds = preds.squeeze(-1).cpu().numpy()
    return preds


def compute_1d_pdp(model, X_bg, feature_idx, grid):
    X_mod = X_bg.copy()
    vals = []
    for v in grid:
        X_mod[:, feature_idx] = v
        vals.append(model_predict(model, X_mod).mean())
    return np.array(vals)


def compute_2d_pdp(model, X_bg, i, j, grid_points=GRID_POINTS_2D):
    x_min, x_max = X_bg[:, i].min(), X_bg[:, i].max()
    y_min, y_max = X_bg[:, j].min(), X_bg[:, j].max()
    xs = np.linspace(x_min, x_max, grid_points)
    ys = np.linspace(y_min, y_max, grid_points)
    X_mod = X_bg.copy()
    Z = np.zeros((grid_points, grid_points), dtype=np.float32)
    for ix, xv in enumerate(xs):
        for iy, yv in enumerate(ys):
            X_mod[:, i] = xv
            X_mod[:, j] = yv
            Z[iy, ix] = model_predict(model, X_mod).mean()
    return xs, ys, Z


def compute_interaction_curves(model, X_bg, var_idx, cond_idx, cond_values,
                                grid_points=GRID_POINTS_1D):
    v_min, v_max = X_bg[:, var_idx].min(), X_bg[:, var_idx].max()
    grid = np.linspace(v_min, v_max, grid_points)
    X_mod = X_bg.copy()
    curves = {}
    for c in cond_values:
        vals = []
        for v in grid:
            X_mod[:, var_idx] = v
            X_mod[:, cond_idx] = c
            vals.append(model_predict(model, X_mod).mean())
        curves[c] = np.array(vals)
    return grid, curves


# ─────────────────────────────────────────────────────────────
# PLOT 1 — 1D PDP Evolution (grid layout)
# ─────────────────────────────────────────────────────────────
def plot_1d_pdp_evolution(models, X_bg, feature_names, true_idx):
    """
    1D PDP for signal features across training epochs.
    Grid: 2 rows if >5 features, else 1 row. Epoch colours from viridis.
    """
    epochs = sorted(models.keys())
    n_true = len(true_idx)
    if n_true == 0:
        print("  [pdp] No true_idx; skipping 1D PDP evolution.")
        return

    nrows = 2 if n_true > 5 else 1
    ncols = (n_true + 1) // 2 if n_true > 5 else n_true
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(3 * ncols, 9), 3.5 * nrows),
        sharey=False,
        constrained_layout=True,
    )
    if n_true == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if n_true > 1 else axes

    cmap = plt.cm.viridis
    epoch_norm = np.linspace(0.15, 0.85, len(epochs)) if len(epochs) > 1 else [0.5]

    fig.suptitle(
        "1D Partial Dependence Plots — Feature Effect Evolution During Training",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax_idx in range(len(true_idx)):
        feat_idx = true_idx[ax_idx]
        ax = axes_flat[ax_idx]
        v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
        grid = np.linspace(v_min, v_max, GRID_POINTS_1D)
        for ei, epoch in enumerate(epochs):
            c = cmap(epoch_norm[ei]) if len(epochs) > 1 else cmap(0.5)
            pdp_vals = compute_1d_pdp(models[epoch], X_bg, feat_idx, grid)
            ax.plot(grid, pdp_vals, color=c, linewidth=1.8, label=f"Epoch {epoch}", alpha=0.9)

        ax.set_xlabel(feature_names[feat_idx], fontsize=10)
        ax.set_title(f"Feature x{feat_idx}", fontsize=10, fontweight="bold", pad=6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if ax_idx % ncols == 0:
            ax.set_ylabel("Avg. Model Output (PDP)", fontsize=10)

        pdp_matrix = np.array([
            compute_1d_pdp(models[e], X_bg, feat_idx, grid) for e in epochs
        ])
        ax.fill_between(grid, pdp_matrix.min(0), pdp_matrix.max(0), alpha=0.10, color="#888888")

    for ax_idx in range(len(true_idx), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    handles = [
        Line2D([0], [0], color=cmap(epoch_norm[ei]), linewidth=2, label=f"Epoch {e}")
        for ei, e in enumerate(epochs)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(epochs), 8), frameon=True)

    out = os.path.join(BASE_OUTPUT_DIR, "pdp_1d_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 2 — 2D PDP Grid (one figure: rows = GT pairs, cols = selected epochs)
# ─────────────────────────────────────────────────────────────
def plot_2d_pdp_grid(models, X_bg, pairs, feature_names):
    """
    Single figure: one row per GT pair, columns = GRID_EPOCHS_2D.
    Shared colour scale per row. Save as pdp_2d_grid.png.
    """
    epochs = [e for e in GRID_EPOCHS_2D if e in models]
    if not epochs or not pairs:
        return
    nrows = len(pairs)
    ncols = len(epochs)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(3 * ncols, 9), 3.5 * nrows),
        constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "2D Partial Dependence — Evolution by Epoch (GT Pairs)",
        fontsize=13, fontweight="bold", y=1.02
    )

    for row, (i, j) in enumerate(pairs):
        print(f"    2D grid row {row + 1}/{nrows} (x{i}, x{j})...", flush=True)
        all_Z = []
        all_xs, all_ys = None, None
        for ei, epoch in enumerate(epochs):
            xs, ys, Z = compute_2d_pdp(models[epoch], X_bg, i, j)
            all_Z.append(Z)
            if all_xs is None:
                all_xs, all_ys = xs, ys
            if (ei + 1) % 2 == 0 or ei == 0:
                print(f"      epoch {epoch} done", flush=True)
        vmin = min(z.min() for z in all_Z)
        vmax = max(z.max() for z in all_Z)
        for col, (epoch, Z) in enumerate(zip(epochs, all_Z)):
            ax = axes[row, col]
            Xg, Yg = np.meshgrid(all_xs, all_ys)
            cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.contour(Xg, Yg, Z, levels=6, colors="white", linewidths=0.4, alpha=0.6)
            ax.set_title(f"Epoch {epoch}", fontsize=10, fontweight="bold")
            ax.set_xlabel(feature_names[i], fontsize=10)
            if col == 0:
                ax.set_ylabel(feature_names[j], fontsize=10)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            fig.colorbar(cf, ax=ax, shrink=0.7, label="Avg output" if col == ncols - 1 else "")

    out = os.path.join(BASE_OUTPUT_DIR, "pdp_2d_grid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 3 — Interaction signature: two pairs side by side (static)
# ─────────────────────────────────────────────────────────────
def plot_interaction_signature(model, X_bg, pair1, pair2, feature_names):
    """
    PDP of first feature conditioned on second for two GT pairs.
    Saves as pdp_interaction_signature.png.
    """
    cond_pct = [0.25, 0.5, 0.75]
    cond_values = [np.percentile(X_bg[:, pair1[1]], p * 100) for p in cond_pct]
    cond_values2 = [np.percentile(X_bg[:, pair2[1]], p * 100) for p in cond_pct]

    grid1, curves1 = compute_interaction_curves(
        model, X_bg, var_idx=pair1[0], cond_idx=pair1[1], cond_values=cond_values
    )
    grid2, curves2 = compute_interaction_curves(
        model, X_bg, var_idx=pair2[0], cond_idx=pair2[1], cond_values=cond_values2
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(
        f"Interaction Signature at Epoch {INTERACTION_EPOCH} — Two GT Pairs",
        fontsize=13, fontweight="bold", y=1.02
    )
    labels = [f"= {c:.2f}" for c in cond_values]
    labels2 = [f"= {c:.2f}" for c in cond_values2]
    linestyles = ["-", "--", ":"]

    ax1 = axes[0]
    for c, col, ls, lab in zip(cond_values, COND_COLORS, linestyles, labels):
        ax1.plot(grid1, curves1[c], color=col, linewidth=2.2, linestyle=ls,
                 label=f"{feature_names[pair1[1]]} {lab}")
    ax1.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="-")
    ax1.set_title(f"Pair ({feature_names[pair1[0]]}, {feature_names[pair1[1]]})", fontsize=10, fontweight="bold")
    ax1.set_xlabel(feature_names[pair1[0]], fontsize=10)
    ax1.set_ylabel("Avg. Model Output (PDP)", fontsize=10)
    ax1.legend(title=f"{feature_names[pair1[1]]} condition", title_fontsize=8)

    ax2 = axes[1]
    for c, col, ls, lab in zip(cond_values2, COND_COLORS, linestyles, labels2):
        ax2.plot(grid2, curves2[c], color=col, linewidth=2.2, linestyle=ls,
                 label=f"{feature_names[pair2[1]]} {lab}")
    ax2.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="-")
    ax2.set_title(f"Pair ({feature_names[pair2[0]]}, {feature_names[pair2[1]]})", fontsize=10, fontweight="bold")
    ax2.set_xlabel(feature_names[pair2[0]], fontsize=10)
    ax2.legend(title=f"{feature_names[pair2[1]]} condition", title_fontsize=8)

    out = os.path.join(BASE_OUTPUT_DIR, "pdp_interaction_signature.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PDP Analysis — Grid Layouts & GIF Animations")
    print("=" * 60)

    ensure_dirs()

    print("\nLoading meta (feature_names, ground_truth)...")
    feature_names, ground_truth = load_meta()
    pairs = ground_truth["pairwise"]
    true_idx = ground_truth["true_idx"]

    print("\nLoading background data...")
    X_bg, _ = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    # Load all snapshot epochs (for grid + GIFs)
    epochs_all = sorted(e for e in SNAPSHOT_EPOCHS if e <= 300)
    print(f"\nLoading model snapshots for {len(epochs_all)} epochs...")
    models = load_unreg_models(epochs_all)

    # 1D PDP evolution (subset of epochs for clear legend)
    epochs_1d = [e for e in SNAPSHOT_EPOCHS_1D if e in models]
    if epochs_1d:
        print("\n[1/3] 1D PDP evolution (signal features)...")
        plot_1d_pdp_evolution(
            {e: models[e] for e in epochs_1d}, X_bg, feature_names, true_idx
        )

    # 2D PDP grid: по одной паре из каждого any_order GT set (макс. 4 строки)
    representative_pairs = []
    seen = set()
    for ao in ground_truth["any_order"]:
        ao_sorted = sorted(ao)
        if len(ao_sorted) >= 2:
            p = (ao_sorted[0], ao_sorted[1])
            if p not in seen:
                representative_pairs.append(p)
                seen.add(p)
    if representative_pairs:
        print(f"[2/3] 2D PDP grid ({len(representative_pairs)} rows × {len(GRID_EPOCHS_2D)} cols)...")
        plot_2d_pdp_grid(models, X_bg, representative_pairs, feature_names)

    # Static interaction signature at final epoch
    if len(pairs) >= 2 and INTERACTION_EPOCH in models:
        print(f"[3/3] Interaction signature (static) at epoch {INTERACTION_EPOCH}...")
        plot_interaction_signature(
            models[INTERACTION_EPOCH], X_bg, pairs[0], pairs[1], feature_names
        )

    print(f"\nDone! All figures saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()