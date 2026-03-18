"""
pdp_analysis.py
========================

Partial Dependence Plot (PDP) analysis for MLP snapshots.
Publication-quality matplotlib styling and static PNG figures.

Run AFTER train_mlp_snapshots.py has generated model snapshots in:
    outputs/snapshots/unregularized/model_epoch_{N}.pt

Generated figures (saved under ./outputs/pdp/static/):
    - pdp_1d_x{i}.png            — 1D PDP per feature (3×3 grid, 9 epochs)
    - pdp_2d_x{i}_x{j}.png       — 2D PDP per feature pair (3×3 grid, 9 epochs)
    - pdp_signature_x{i}_x{j}.png — interaction signature per pair (3×3 grid, 9 epochs)
    (plot_1d_pdp_evolution, plot_2d_pdp_grid, plot_interaction_signature still in code, not called from main)

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

# Inference on this repo is kept on CPU for speed/consistency.
DEVICE = torch.device("cpu")

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
    "figure.dpi":         220,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})

COND_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # for conditioned curves

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SNAPSHOT_DIR       = "./outputs/snapshots"       # unregularized/, l2/ live here
UNREG_SNAPSHOT_DIR = os.path.join(SNAPSHOT_DIR, "unregularized")
PDP_STATIC_DIR     = "./outputs/pdp/static"      # all PNG from pdp_analysis.py
PDP_GIFS_DIR       = "./outputs/pdp/gifs"        # all GIF from pdp_gifs.py

# 9 epochs → 3×3 grid for ALL static plots (1D PDP, 2D PDP, interaction signature)
GRID_EPOCHS = [3, 10, 20, 30, 40, 50, 100, 150, 220]
INTERACTION_EPOCH  = 300

GRID_POINTS_1D = 60
GRID_POINTS_2D = 30   # 30×30 быстрее 40×40, картинка чуть грубее
MC_MAX_SAMPLES = 600  # меньше сэмплов = быстрее 2D PDP и GIF
GIF_DPI = 100
GIF_FPS = 4
GIF_INTERVAL_MS = 250

# Larger static figures improve readability for 3×3 subplot grids.
STATIC_FIG_DPI = 220
GRID_FIGSIZE_3X3 = (18, 15)

# True noise features per synth function (0-indexed).
# These are features that do NOT appear anywhere in the formula,
# not just features absent from interaction sets.
# ground_truth["noise_idx"] is interaction-based and includes main-effect-only
# features, which is wrong for PDP overfitting analysis.
TRUE_NOISE_IDX = {
    0: [5],           # F1: x6 not in formula
    1: [5],           # F2: x6 not in formula
    2: [5],           # F3: x6 not in formula
    3: [5],           # F4: x6 not in formula
    4: [],            # F5: all features in formula
    5: [6],           # F6: x7 not in formula
    6: [],            # F7: all features in formula (np.sum(X))
    7: [],            # F8: all features in formula
    8: [],            # F9: all features in formula
    9: [5, 7, 9],     # F10: x6, x8, x10 not in formula
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def all_feature_pairs(n_features: int):
    """All unique feature pairs (i < j)."""
    return [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]


def ensure_dirs():
    os.makedirs(PDP_STATIC_DIR, exist_ok=True)
    os.makedirs(PDP_GIFS_DIR, exist_ok=True)
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

    out = os.path.join(PDP_STATIC_DIR, "pdp_1d_evolution.png")
    fig.savefig(out, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 1b — 1D PDP Individual (one PNG per feature × epochs)
# ─────────────────────────────────────────────────────────────
def plot_1d_pdp_individual(models, X_bg, feature_names):
    """
    For each feature (all), save one PNG: 3×3 grid, 9 epochs (GRID_EPOCHS).
    Each subplot: 1D PDP curve at that epoch. Shared Y-axis across all 9.
    File: pdp_1d_x{i}.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    if not epochs:
        return
    n_feat = len(feature_names)
    for feat_idx in range(n_feat):
        print(f"    1D PDP individual {feat_idx + 1}/{n_feat} (x{feat_idx})...", flush=True)
        v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
        grid = np.linspace(v_min, v_max, GRID_POINTS_1D)
        all_vals = []
        for epoch in epochs:
            pdp_vals = compute_1d_pdp(models[epoch], X_bg, feat_idx, grid)
            all_vals.append(pdp_vals)
        all_vals = np.array(all_vals)
        y_min, y_max = all_vals.min(), all_vals.max()
        fig, axes = plt.subplots(
            3, 3,
            figsize=GRID_FIGSIZE_3X3,
            sharey=True,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        fig.suptitle(
            f"1D PDP — x{feat_idx} ({feature_names[feat_idx]}) — Evolution by Epoch",
            fontsize=13, fontweight="bold", y=1.02,
        )
        for idx, (epoch, pdp_vals) in enumerate(zip(epochs, all_vals)):
            ax = axes_flat[idx]
            ax.plot(grid, pdp_vals, color="#2ca02c", linewidth=1.8)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"Epoch {epoch}", fontsize=10, fontweight="bold")
            ax.set_xlabel(feature_names[feat_idx], fontsize=10)
            if idx % 3 == 0:
                ax.set_ylabel("Avg. Model Output (PDP)", fontsize=10)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        out = os.path.join(PDP_STATIC_DIR, f"pdp_1d_x{feat_idx}.png")
        fig.savefig(out, dpi=STATIC_FIG_DPI)
        plt.close(fig)
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 2 — 2D PDP Grid (one figure: rows = GT pairs, cols = selected epochs)
# ─────────────────────────────────────────────────────────────
def plot_2d_pdp_grid(models, X_bg, pairs, feature_names):
    """
    Single figure: one row per GT pair, columns = GRID_EPOCHS.
    Shared colour scale per row. Save as pdp_2d_grid.png.
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
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
        "2D Partial Dependence — Evolution by Epoch",
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

    out = os.path.join(PDP_STATIC_DIR, "pdp_2d_grid.png")
    fig.savefig(out, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 2b — 2D PDP Individual (one PNG per GT pair, dense epochs)
# ─────────────────────────────────────────────────────────────
def plot_2d_pdp_individual(models, X_bg, pairs, feature_names):
    """
    For each feature pair, save a separate PNG: 3×3 grid, 9 epochs (GRID_EPOCHS).
    Shared colour scale (global vmin/vmax). File: pdp_2d_x{i}_x{j}.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    if not epochs or not pairs:
        return
    for pair_idx, (i, j) in enumerate(pairs):
        print(f"    2D PDP individual {pair_idx + 1}/{len(pairs)} (x{i}, x{j})...", flush=True)
        all_Z = []
        all_xs, all_ys = None, None
        for epoch in epochs:
            xs, ys, Z = compute_2d_pdp(models[epoch], X_bg, i, j)
            all_Z.append(Z)
            if all_xs is None:
                all_xs, all_ys = xs, ys
        vmin = min(z.min() for z in all_Z)
        vmax = max(z.max() for z in all_Z)
        fig, axes = plt.subplots(
            3, 3,
            figsize=GRID_FIGSIZE_3X3,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        fig.suptitle(
            f"2D PDP — (x{i}, x{j}) — Evolution by Epoch",
            fontsize=13, fontweight="bold", y=1.02,
        )
        for idx, (epoch, Z) in enumerate(zip(epochs, all_Z)):
            ax = axes_flat[idx]
            Xg, Yg = np.meshgrid(all_xs, all_ys)
            cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.contour(Xg, Yg, Z, levels=6, colors="white", linewidths=0.4, alpha=0.6)
            ax.set_title(f"Epoch {epoch}", fontsize=10, fontweight="bold")
            ax.set_xlabel(feature_names[i], fontsize=10)
            if idx % 3 == 0:
                ax.set_ylabel(feature_names[j], fontsize=10)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            if idx == 8:
                fig.colorbar(cf, ax=ax, shrink=0.7, label="Avg output")
        out = os.path.join(PDP_STATIC_DIR, f"pdp_2d_x{i}_x{j}.png")
        fig.savefig(out, dpi=STATIC_FIG_DPI)
        plt.close(fig)
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 3 — Interaction signature: two pairs side by side (static)
# ─────────────────────────────────────────────────────────────
def plot_interaction_signature(model, X_bg, pair1, pair2, feature_names):
    """
    PDP of first feature conditioned on second for two pairs.
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

    out = os.path.join(PDP_STATIC_DIR, "pdp_interaction_signature.png")
    fig.savefig(out, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 3b — Interaction signature individual (one PNG per GT pair × epochs)
# ─────────────────────────────────────────────────────────────
def plot_interaction_signatures_individual(models, X_bg, pairs, feature_names):
    """
    For each pair (i, j), save one PNG: 3×3 grid, 9 epochs (GRID_EPOCHS).
    Each subplot: 3 conditioned PDP curves (25th/50th/75th percentile).
    File: pdp_signature_x{i}_x{j}.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    if not epochs or not pairs:
        return
    cond_pct = [0.25, 0.5, 0.75]
    linestyles = ["-", "--", ":"]

    for pair_idx, (i, j) in enumerate(pairs):
        print(f"    Interaction signature {pair_idx + 1}/{len(pairs)} (x{i}, x{j})...", flush=True)
        cond_values = [np.percentile(X_bg[:, j], p * 100) for p in cond_pct]

        # PASS 1: precompute all epochs + global y-limits
        precomputed = []
        all_curve_vals = []
        for epoch in epochs:
            model = models[epoch]
            grid, curves = compute_interaction_curves(
                model, X_bg, var_idx=i, cond_idx=j, cond_values=cond_values
            )
            precomputed.append((epoch, grid, curves))
            for c in cond_values:
                all_curve_vals.append(np.asarray(curves[c], dtype=float).ravel())

        flat = np.concatenate(all_curve_vals) if all_curve_vals else np.array([0.0])
        global_ymin = float(flat.min())
        global_ymax = float(flat.max())
        pad = (global_ymax - global_ymin) * 0.05
        global_ymin -= pad
        global_ymax += pad

        # PASS 2: plot with shared y-limits
        fig, axes = plt.subplots(
            3, 3,
            figsize=GRID_FIGSIZE_3X3,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        fig.suptitle(
            f"Interaction Signature — (x{i}, x{j}) — Evolution by Epoch",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for idx, (epoch, grid, curves) in enumerate(precomputed):
            ax = axes_flat[idx]
            for c, col_c, ls in zip(cond_values, COND_COLORS, linestyles):
                ax.plot(grid, curves[c], color=col_c, linewidth=2.2, linestyle=ls)
            ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="-")
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_title(f"Epoch {epoch}", fontsize=10, fontweight="bold")
            ax.set_xlabel(feature_names[i], fontsize=10)
            if idx % 3 == 0:
                ax.set_ylabel("Avg. Model Output (PDP)", fontsize=10)

        for idx in range(len(precomputed), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        out = os.path.join(PDP_STATIC_DIR, f"pdp_signature_x{i}_x{j}.png")
        fig.savefig(out, dpi=STATIC_FIG_DPI)
        plt.close(fig)
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# OVERFITTING DETECTION (Area 1) — PDP Roughness & Range
# ─────────────────────────────────────────────────────────────
def compute_pdp_roughness(pdp_values):
    """
    Roughness = sum of squared second differences (discrete 2nd derivative).
    Measures how "wiggly" a 1D PDP curve is; noise features get high roughness when overfitting.
    """
    pdp = np.asarray(pdp_values, dtype=np.float64)
    n = len(pdp)
    if n < 3:
        return 0.0
    second_diff = pdp[2:] - 2 * pdp[1:-1] + pdp[:-2]
    return float(np.sum(second_diff ** 2))


def plot_pdp_roughness_evolution(models, X_bg, feature_names, true_idx, noise_idx):
    """
    PDP roughness vs epoch for signal vs noise features.
    Signal = blue solid; noise = red dashed. Bold lines for mean signal / mean noise.
    Saves to pdp_roughness_evolution.png in PDP_STATIC_DIR.
    """
    epochs = sorted(models.keys())
    if not epochs:
        return
    n_feat = len(feature_names)
    # roughness[epoch_idx][feat_idx]
    roughness = np.zeros((len(epochs), n_feat))
    for ei, epoch in enumerate(epochs):
        model = models[epoch]
        for feat_idx in range(n_feat):
            v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
            grid = np.linspace(v_min, v_max, GRID_POINTS_1D)
            pdp_vals = compute_1d_pdp(model, X_bg, feat_idx, grid)
            roughness[ei, feat_idx] = compute_pdp_roughness(pdp_vals)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDP Roughness")
    ax.set_title("PDP Roughness Evolution — Signal vs Noise Features (Overfitting Detection)")
    ax.set_xlim(epochs[0], epochs[-1] * 1.15)

    for feat_idx in true_idx:
        vals = roughness[:, feat_idx]
        ax.plot(epochs, vals, color="#377eb8", linestyle="-", linewidth=1.2, alpha=0.8)
        ax.annotate(
            feature_names[feat_idx],
            xy=(epochs[-1], vals[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color="#377eb8",
            va="center",
        )
    for feat_idx in noise_idx:
        vals = roughness[:, feat_idx]
        ax.plot(epochs, vals, color="#e41a1c", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(
            feature_names[feat_idx],
            xy=(epochs[-1], vals[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color="#e41a1c",
            va="center",
        )

    if true_idx:
        mean_signal = roughness[:, true_idx].mean(axis=1)
        ax.plot(epochs, mean_signal, color="#377eb8", linestyle="-", linewidth=3)
    if noise_idx:
        mean_noise = roughness[:, noise_idx].mean(axis=1)
        ax.plot(epochs, mean_noise, color="#e41a1c", linestyle="--", linewidth=3)

    handles = [
        Line2D([0], [0], color="#377eb8", linestyle="-", linewidth=1.2, label="Signal features"),
        Line2D([0], [0], color="#e41a1c", linestyle="--", linewidth=1.2, label="Noise features"),
        Line2D([0], [0], color="#377eb8", linestyle="-", linewidth=3, label="Mean (signal)"),
        Line2D([0], [0], color="#e41a1c", linestyle="--", linewidth=3, label="Mean (noise)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    out = os.path.join(PDP_STATIC_DIR, "pdp_roughness_evolution.png")
    fig.savefig(out, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pdp_range_evolution(models, X_bg, feature_names, true_idx, noise_idx):
    """
    PDP range (max - min) vs epoch — "Feature Importance Inflation" for overfitting.
    Signal = blue solid; noise = red dashed. Bold lines for mean signal / mean noise.
    Saves to pdp_range_evolution.png in PDP_STATIC_DIR.
    """
    epochs = sorted(models.keys())
    if not epochs:
        return
    n_feat = len(feature_names)
    pdp_range = np.zeros((len(epochs), n_feat))
    for ei, epoch in enumerate(epochs):
        model = models[epoch]
        for feat_idx in range(n_feat):
            v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
            grid = np.linspace(v_min, v_max, GRID_POINTS_1D)
            pdp_vals = compute_1d_pdp(model, X_bg, feat_idx, grid)
            pdp_range[ei, feat_idx] = float(np.max(pdp_vals) - np.min(pdp_vals))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PDP Range (max − min)")
    ax.set_title("PDP Range Evolution — Signal vs Noise Features (Overfitting Detection)")
    ax.set_xlim(epochs[0], epochs[-1] * 1.15)

    for feat_idx in true_idx:
        vals = pdp_range[:, feat_idx]
        ax.plot(epochs, vals, color="#377eb8", linestyle="-", linewidth=1.2, alpha=0.8)
        ax.annotate(
            feature_names[feat_idx],
            xy=(epochs[-1], vals[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color="#377eb8",
            va="center",
        )
    for feat_idx in noise_idx:
        vals = pdp_range[:, feat_idx]
        ax.plot(epochs, vals, color="#e41a1c", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.annotate(
            feature_names[feat_idx],
            xy=(epochs[-1], vals[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color="#e41a1c",
            va="center",
        )

    if true_idx:
        mean_signal = pdp_range[:, true_idx].mean(axis=1)
        ax.plot(epochs, mean_signal, color="#377eb8", linestyle="-", linewidth=3)
    if noise_idx:
        mean_noise = pdp_range[:, noise_idx].mean(axis=1)
        ax.plot(epochs, mean_noise, color="#e41a1c", linestyle="--", linewidth=3)

    handles = [
        Line2D([0], [0], color="#377eb8", linestyle="-", linewidth=1.2, label="Signal features"),
        Line2D([0], [0], color="#e41a1c", linestyle="--", linewidth=1.2, label="Noise features"),
        Line2D([0], [0], color="#377eb8", linestyle="-", linewidth=3, label="Mean (signal)"),
        Line2D([0], [0], color="#e41a1c", linestyle="--", linewidth=3, label="Mean (noise)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=9)
    out = os.path.join(PDP_STATIC_DIR, "pdp_range_evolution.png")
    fig.savefig(out, dpi=STATIC_FIG_DPI)
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
    # 2D PDP + signatures: by default generate for ALL pairs (i < j).
    pairs = all_feature_pairs(len(feature_names))
    true_idx = ground_truth["true_idx"]
    # Overfitting plots use formula-based noise; rest of script uses ground_truth as-is.
    noise_idx = TRUE_NOISE_IDX.get(SYNTH_FN_IDX, ground_truth.get("noise_idx", []))
    true_idx_for_overfit = [i for i in range(len(feature_names)) if i not in noise_idx]

    print("\nLoading background data...")
    X_bg, _ = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    # Load all snapshot epochs (for grid + GIFs)
    epochs_all = sorted(e for e in SNAPSHOT_EPOCHS if e <= 300)
    print(f"\nLoading model snapshots for {len(epochs_all)} epochs...")
    models = load_unreg_models(epochs_all)

    # 1D PDP individual: one PNG per feature (all 10), 1 row × PDP_2D_EPOCHS cols
    print("\n[1/5] 1D PDP individual (all features)...")
    plot_1d_pdp_individual(models, X_bg, feature_names)

    # 2D PDP individual: one PNG per GT pair, 3×3 grid (GRID_EPOCHS)
    if pairs:
        print(f"[2/5] 2D PDP individual ({len(pairs)} figures, 3×3 grid)...")
        plot_2d_pdp_individual(models, X_bg, pairs, feature_names)

    # Interaction signature individual: one PNG per GT pair (3×3 grid)
    if pairs:
        print(f"[3/5] Interaction signatures individual ({len(pairs)} figures)...")
        plot_interaction_signatures_individual(models, X_bg, pairs, feature_names)

    # Overfitting detection: PDP roughness evolution (signal vs noise)
    print("\n[4/5] PDP roughness evolution (overfitting detection)...")
    plot_pdp_roughness_evolution(models, X_bg, feature_names, true_idx_for_overfit, noise_idx)

    # Overfitting detection: PDP range evolution (feature importance inflation)
    print("\n[5/5] PDP range evolution (overfitting detection)...")
    plot_pdp_range_evolution(models, X_bg, feature_names, true_idx_for_overfit, noise_idx)

    print(f"\nDone! All figures saved to: {PDP_STATIC_DIR}")


if __name__ == "__main__":
    main()