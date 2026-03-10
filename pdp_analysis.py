"""
pdp_analysis.py
========================

Partial Dependence Plot (PDP) analysis for MLP snapshots.
Improved version with publication-quality matplotlib styling.

Run AFTER train_mlp_snapshots.py has generated model snapshots in:
    outputs/snapshots/unregularized/model_epoch_{N}.pt

Generated figures (saved under ./outputs/snapshots/):
    - pdp_1d_evolution.png          — 1D PDP curves for true features across epochs
    - pdp_2d_modifier.png           — 2D heatmap: modifier interaction (x0, x1)
    - pdp_2d_crossover.png          — 2D heatmap: crossover interaction (x2, x3)
    - pdp_modifier_vs_crossover.png — side-by-side interaction signature comparison
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from train_mlp_snapshots import generate_data, MLP, N_FEATURES, SEED


# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE — academic / publication look
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.titleweight":   "bold",
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

# Colour palettes
EPOCH_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]  # red→purple
COND_COLORS  = ["#e41a1c", "#377eb8", "#4daf4a"]   # red / blue / green


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR    = "./outputs/snapshots"
UNREG_SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "unregularized")

SNAPSHOT_EPOCHS_1D  = [1, 10, 50, 100, 300]
SNAPSHOT_EPOCHS_2D  = [1, 50, 300]
INTERACTION_EPOCH   = 300

GRID_POINTS_1D = 60
GRID_POINTS_2D = 40
MC_MAX_SAMPLES = 1000


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


def load_unreg_models(epochs):
    models = {}
    for epoch in epochs:
        path = os.path.join(UNREG_SNAPSHOT_DIR, f"model_epoch_{epoch}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing snapshot: {path}")
        model = MLP(N_FEATURES)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        models[epoch] = model
        print(f"  Loaded epoch {epoch}")
    return models


def get_background_data(mc_max_samples=MC_MAX_SAMPLES):
    data = generate_data(n_samples=3000, n_true=5, n_noise=5, seed=SEED)
    X = data["X"]
    feature_names = data["feature_names"]
    rng = np.random.RandomState(SEED)
    if X.shape[0] > mc_max_samples:
        idx = rng.choice(X.shape[0], mc_max_samples, replace=False)
        X = X[idx]
    return X, feature_names


def model_predict(model, X):
    with torch.no_grad():
        preds = model(torch.from_numpy(X.astype(np.float32))).cpu().numpy()
    return preds


def compute_1d_pdp(model, X_bg, feature_idx, grid):
    X_mod = X_bg.copy()
    vals = []
    for v in grid:
        X_mod[:, feature_idx] = v
        vals.append(model_predict(model, X_mod).mean())
    return np.array(vals)


def compute_2d_pdp(model, X_bg, i, j, grid_points=GRID_POINTS_2D):
    xs = np.linspace(0.0, 1.0, grid_points)
    ys = np.linspace(0.0, 1.0, grid_points)
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
    grid = np.linspace(0.0, 1.0, grid_points)
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
# PLOT 1 — 1D PDP Evolution
# ─────────────────────────────────────────────────────────────
def plot_1d_pdp_evolution(models, X_bg, feature_names):
    """
    1D PDP for true features (x0–x4) across training epochs.
    Each subplot shows how the model's perception of a feature evolves.
    """
    epochs  = sorted(models.keys())
    grid    = np.linspace(0.0, 1.0, GRID_POINTS_1D)
    n_true  = 5

    fig, axes = plt.subplots(1, n_true, figsize=(14, 3.6), sharey=False)
    fig.suptitle(
        "1D Partial Dependence Plots — Feature Effect Evolution During Training",
        fontsize=12, fontweight="bold", y=1.01
    )

    for feat_idx in range(n_true):
        ax = axes[feat_idx]
        for color, epoch in zip(EPOCH_COLORS, epochs):
            pdp_vals = compute_1d_pdp(models[epoch], X_bg, feat_idx, grid)
            ax.plot(grid, pdp_vals, color=color, linewidth=1.8,
                    label=f"Epoch {epoch}", alpha=0.9)

        # Feature-specific annotations
        ax.set_xlabel(feature_names[feat_idx], fontsize=10)
        ax.set_title(f"Feature x{feat_idx}", pad=6)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        if feat_idx == 0:
            ax.set_ylabel("Avg. Model Output (PDP)", fontsize=10)

        # Light shading between min/max across epochs to show uncertainty band
        pdp_matrix = np.array([
            compute_1d_pdp(models[e], X_bg, feat_idx, grid) for e in epochs
        ])
        ax.fill_between(grid, pdp_matrix.min(0), pdp_matrix.max(0),
                         alpha=0.10, color="#888888")

    # Shared legend below all subplots
    handles = [
        Line2D([0], [0], color=c, linewidth=2, label=f"Epoch {e}")
        for c, e in zip(EPOCH_COLORS, epochs)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(epochs),
               bbox_to_anchor=(0.5, -0.08), frameon=True)

    fig.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, "pdp_1d_evolution.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 2 — 2D PDP: Modifier Interaction (x0, x1)
# ─────────────────────────────────────────────────────────────
def plot_2d_pdp_modifier(models, X_bg):
    """
    2D PDP heatmaps for modifier interaction (x0 * x1).
    Modifier: magnitude changes but sign stays constant.
    Expected: gradient intensifies toward top-right, no sign flip.
    """
    epochs = SNAPSHOT_EPOCHS_2D
    fig, axes = plt.subplots(1, len(epochs), figsize=(13, 4))
    fig.suptitle(
        "2D PDP — Modifier Interaction (x0 × x1)\n"
        r"Signal: $y \supset x_0 \cdot x_1$   |   "
        "Expect: smooth gradient, no sign reversal",
        fontsize=11, fontweight="bold"
    )

    vmin, vmax = None, None
    # First pass to find global colour scale
    all_Z = []
    for epoch in epochs:
        _, _, Z = compute_2d_pdp(models[epoch], X_bg, i=0, j=1)
        all_Z.append(Z)
    vmin = min(z.min() for z in all_Z)
    vmax = max(z.max() for z in all_Z)

    for ax, epoch, Z in zip(axes, epochs, all_Z):
        xs = np.linspace(0, 1, GRID_POINTS_2D)
        ys = np.linspace(0, 1, GRID_POINTS_2D)
        Xg, Yg = np.meshgrid(xs, ys)
        cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="YlOrRd",
                          vmin=vmin, vmax=vmax)
        ax.contour(Xg, Yg, Z, levels=6, colors="white",
                   linewidths=0.5, alpha=0.6)
        ax.set_title(f"Epoch {epoch}", fontsize=11)
        ax.set_xlabel("x0  (modifier feature)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("x1", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        fig.colorbar(cf, ax=ax, shrink=0.85, label="Avg output" if ax is axes[-1] else "")

    fig.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, "pdp_2d_modifier.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 3 — 2D PDP: Crossover Interaction (x2, x3)
# ─────────────────────────────────────────────────────────────
def plot_2d_pdp_crossover(models, X_bg):
    """
    2D PDP heatmaps for crossover interaction x2*(x3 - 0.5).
    Crossover: effect of x2 is negative when x3 < 0.5 and positive when x3 > 0.5.
    Expected: blue (negative) below x3=0.5, red (positive) above — classic sign reversal.
    """
    epochs = SNAPSHOT_EPOCHS_2D
    fig, axes = plt.subplots(1, len(epochs), figsize=(13, 4))
    fig.suptitle(
        "2D PDP — Crossover Interaction (x2, x3)\n"
        r"Signal: $y \supset x_2 \cdot (x_3 - 0.5)$   |   "
        "Expect: sign reversal at x3 = 0.5",
        fontsize=11, fontweight="bold"
    )

    all_Z = []
    for epoch in epochs:
        _, _, Z = compute_2d_pdp(models[epoch], X_bg, i=2, j=3)
        all_Z.append(Z)

    # Diverging colour centred at 0
    abs_max = max(abs(z).max() for z in all_Z)

    for ax, epoch, Z in zip(axes, epochs, all_Z):
        xs = np.linspace(0, 1, GRID_POINTS_2D)
        ys = np.linspace(0, 1, GRID_POINTS_2D)
        Xg, Yg = np.meshgrid(xs, ys)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        cf = ax.contourf(Xg, Yg, Z, levels=20, cmap="RdBu_r", norm=norm)
        ax.contour(Xg, Yg, Z, levels=[0], colors="black",
                   linewidths=1.5, linestyles="--")
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":",
                   alpha=0.7, label="x3 = 0.5")
        ax.set_title(f"Epoch {epoch}", fontsize=11)
        ax.set_xlabel("x2", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("x3  (crossover feature)", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        fig.colorbar(cf, ax=ax, shrink=0.85,
                     label="Avg output" if ax is axes[-1] else "")

    # Annotate sign regions on the last panel
    axes[-1].text(0.5, 0.25, "Negative\neffect", ha="center", va="center",
                  fontsize=8, color="#1a5276",
                  transform=axes[-1].transAxes,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    axes[-1].text(0.5, 0.72, "Positive\neffect", ha="center", va="center",
                  fontsize=8, color="#922b21",
                  transform=axes[-1].transAxes,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, "pdp_2d_crossover.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# PLOT 4 — Modifier vs Crossover Signature
# ─────────────────────────────────────────────────────────────
def plot_modifier_vs_crossover_signature(model, X_bg):
    """
    The key diagnostic plot.
    Left : PDP(x0) conditioned on x1 → lines fan but DON'T cross  (Modifier)
    Right: PDP(x2) conditioned on x3 → lines CROSS each other     (Crossover)
    """
    cond_values = [0.1, 0.5, 0.9]
    grid_x0, modifier_curves = compute_interaction_curves(
        model, X_bg, var_idx=0, cond_idx=1, cond_values=cond_values
    )
    grid_x2, crossover_curves = compute_interaction_curves(
        model, X_bg, var_idx=2, cond_idx=3, cond_values=cond_values
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Interaction Signature at Epoch {INTERACTION_EPOCH} — Modifier vs. Crossover",
        fontsize=12, fontweight="bold"
    )

    labels = [f"= {c:.1f}" for c in cond_values]
    linestyles = ["-", "--", ":"]

    # ── Left: Modifier ─────────────────────────────────────────
    ax_mod = axes[0]
    for c, col, ls, lab in zip(cond_values, COND_COLORS, linestyles, labels):
        ax_mod.plot(grid_x0, modifier_curves[c], color=col, linewidth=2.2,
                    linestyle=ls, label=f"x1 {lab}")

    ax_mod.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="-")
    ax_mod.set_title("Modifier Interaction\n(x0 × x1)", fontweight="bold")
    ax_mod.set_xlabel("x0 value", fontsize=10)
    ax_mod.set_ylabel("Avg. Model Output (PDP)", fontsize=10)
    ax_mod.legend(title="x1 condition", title_fontsize=8)
    ax_mod.annotate(
        "Lines fan out\nbut do NOT cross\n→ sign constant",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8dc",
                  edgecolor="#c8a800", alpha=0.9)
    )

    # ── Right: Crossover ───────────────────────────────────────
    ax_cross = axes[1]
    for c, col, ls, lab in zip(cond_values, COND_COLORS, linestyles, labels):
        ax_cross.plot(grid_x2, crossover_curves[c], color=col, linewidth=2.2,
                      linestyle=ls, label=f"x3 {lab}")

    ax_cross.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="-")
    ax_cross.set_title("Crossover Interaction\n(x2 × (x3 − 0.5))", fontweight="bold")
    ax_cross.set_xlabel("x2 value", fontsize=10)
    ax_cross.legend(title="x3 condition", title_fontsize=8)
    ax_cross.annotate(
        "Lines CROSS each other\n→ sign reverses at x3 = 0.5",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fde8e8",
                  edgecolor="#c0392b", alpha=0.9)
    )

    # Mark the crossing zone on the crossover panel
    # Find approximate crossing x-coordinate
    diff_01 = crossover_curves[0.1] - crossover_curves[0.9]
    sign_changes = np.where(np.diff(np.sign(diff_01)))[0]
    if len(sign_changes):
        cross_x = grid_x2[sign_changes[0]]
        ax_cross.axvline(cross_x, color="#555555", linewidth=1.2,
                         linestyle="--", alpha=0.6)
        ax_cross.text(cross_x + 0.02, ax_cross.get_ylim()[0] * 0.9,
                      f"x≈{cross_x:.2f}", fontsize=7.5, color="#555555")

    fig.tight_layout()
    out = os.path.join(BASE_OUTPUT_DIR, "pdp_modifier_vs_crossover.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PDP Analysis — Improved Publication-Quality Plots")
    print("=" * 60)

    ensure_dirs()

    print("\nLoading background data...")
    X_bg, feature_names = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    epochs_needed = sorted(
        set(SNAPSHOT_EPOCHS_1D) | set(SNAPSHOT_EPOCHS_2D) | {INTERACTION_EPOCH}
    )
    print(f"\nLoading model snapshots for epochs: {epochs_needed}")
    models = load_unreg_models(epochs_needed)

    print("\n[1/4] 1D PDP evolution...")
    plot_1d_pdp_evolution(
        {e: models[e] for e in SNAPSHOT_EPOCHS_1D}, X_bg, feature_names
    )

    print("[2/4] 2D PDP — Modifier interaction...")
    plot_2d_pdp_modifier(models, X_bg)

    print("[3/4] 2D PDP — Crossover interaction...")
    plot_2d_pdp_crossover(models, X_bg)

    print(f"[4/4] Interaction signature at epoch {INTERACTION_EPOCH}...")
    plot_modifier_vs_crossover_signature(models[INTERACTION_EPOCH], X_bg)

    print(f"\nDone! All figures saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()