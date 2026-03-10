"""
ice_hstat_analysis.py
========================

ICE (Individual Conditional Expectation) plots and
Friedman's H-statistic interaction analysis for MLP snapshots.

Run AFTER train_mlp_snapshots.py has generated unregularized model snapshots in:
    ./outputs/snapshots/unregularized/model_epoch_{N}.pt

All figures are saved under:
    ./outputs/snapshots/
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

from train_mlp_snapshots import generate_data, MLP, N_FEATURES, SEED


# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE — academic / publication look (match PDP script)
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


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = "./outputs/snapshots"
UNREG_SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "unregularized")

EPOCHS_ALL: List[int] = [1, 10, 50, 100, 300]
EPOCHS_ICE: List[int] = [1, 50, 300]
EPOCH_HSTAT_300: int = 300

GRID_POINTS_1D: int = 60
GRID_POINTS_2D: int = 20  # coarser grid for H-statistic, as requested
MC_MAX_SAMPLES: int = 1000


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    if not os.path.isdir(UNREG_SNAPSHOT_DIR):
        raise FileNotFoundError(
            f"Snapshot directory not found: {UNREG_SNAPSHOT_DIR}\n"
            "Run train_mlp_snapshots.py first to generate model snapshots."
        )


def load_unreg_models(epochs: List[int]) -> Dict[int, MLP]:
    """
    Load unregularized models for the specified epochs.
    """
    models: Dict[int, MLP] = {}
    for epoch in epochs:
        path = os.path.join(UNREG_SNAPSHOT_DIR, f"model_epoch_{epoch}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing snapshot: {path}")
        model = MLP(N_FEATURES)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models[epoch] = model
        print(f"  Loaded model snapshot for epoch {epoch}")
    return models


def get_background_data(mc_max_samples: int = MC_MAX_SAMPLES) -> Tuple[np.ndarray, List[str]]:
    """
    Regenerate the same synthetic dataset and subsample a background
    set for Monte Carlo ICE / PDP / H-statistic computation.
    """
    data = generate_data(n_samples=3000, n_true=5, n_noise=5, seed=SEED)
    X = data["X"]
    feature_names = data["feature_names"]

    n_samples = X.shape[0]
    if n_samples > mc_max_samples:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n_samples, size=mc_max_samples, replace=False)
        X = X[idx]
    return X, feature_names


def model_predict(model: MLP, X: np.ndarray) -> np.ndarray:
    """
    Run PyTorch model on NumPy array and return predictions as NumPy.
    """
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32))
        preds = model(x_t).cpu().numpy()
    return preds


# ─────────────────────────────────────────────────────────────
# CORE COMPUTATIONS: ICE, PDP, 2D PDP, H-statistic
# ─────────────────────────────────────────────────────────────
def compute_ice(model: MLP,
                X_bg: np.ndarray,
                feature_idx: int,
                grid: np.ndarray) -> np.ndarray:
    """
    For each sample in X_bg, compute predictions across the grid
    while holding all other features fixed at that sample's values.
    Returns ice_curves: np.ndarray shape [n_samples, len(grid)].
    """
    ice_curves = []
    for i in range(len(X_bg)):
        # Repeat this sample along the grid dimension
        X_rep = np.tile(X_bg[i], (len(grid), 1))
        X_rep[:, feature_idx] = grid
        preds = model_predict(model, X_rep)
        ice_curves.append(preds)
    return np.array(ice_curves)  # [n_samples, len(grid)]


def cluster_ice_by_slope(ice_curves: np.ndarray,
                         grid: np.ndarray,
                         n_sample: int = 30,
                         seed: int = 42):
    """
    Classify each ICE curve by its slope (linear regression over grid).
    Returns three groups of indices (sampled within each cluster) and
    the full array of slopes for diagnostics / histograms.
    """
    slopes = []
    for curve in ice_curves:
        # fit line: slope = cov(grid, curve) / var(grid)
        slope = np.polyfit(grid, curve, 1)[0]
        slopes.append(slope)
    slopes = np.array(slopes)

    threshold = 0.3 * slopes.std()
    pos_idx = np.where(slopes > threshold)[0]
    neu_idx = np.where(np.abs(slopes) <= threshold)[0]
    neg_idx = np.where(slopes < -threshold)[0]

    rng = np.random.RandomState(seed)

    def sample(idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return idx
        return rng.choice(idx, size=min(n_sample, len(idx)), replace=False)

    return sample(pos_idx), sample(neu_idx), sample(neg_idx), slopes


def compute_1d_pdp(model: MLP,
                   X_bg: np.ndarray,
                   feature_idx: int,
                   grid: np.ndarray) -> np.ndarray:
    """
    Monte Carlo 1D PDP for a single feature.
    """
    X_mod = X_bg.copy()
    vals: List[float] = []
    for v in grid:
        X_mod[:, feature_idx] = v
        vals.append(float(model_predict(model, X_mod).mean()))
    return np.array(vals)


def compute_2d_pdp(model: MLP,
                   X_bg: np.ndarray,
                   i: int,
                   j: int,
                   grid_points: int = GRID_POINTS_2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo 2D PDP for a feature pair (i, j).
    Returns xs, ys, Z with shape [grid_points] x [grid_points].
    """
    xs = np.linspace(0.0, 1.0, grid_points)
    ys = np.linspace(0.0, 1.0, grid_points)
    X_mod = X_bg.copy()
    Z = np.zeros((grid_points, grid_points), dtype=np.float32)

    for ix, xv in enumerate(xs):
        for iy, yv in enumerate(ys):
            X_mod[:, i] = xv
            X_mod[:, j] = yv
            Z[iy, ix] = float(model_predict(model, X_mod).mean())

    return xs, ys, Z


def compute_hstat(model: MLP,
                  X_bg: np.ndarray,
                  i: int,
                  j: int,
                  grid_points: int = GRID_POINTS_2D) -> float:
    """
    Compute Friedman H-statistic for feature pair (i, j).

    H²(i,j) = Σ [PDP_ij(xi, xj) - PDP_i(xi) - PDP_j(xj)]²
               ÷ Σ [PDP_ij(xi, xj)]²
    """
    xs, ys, pdp_ij = compute_2d_pdp(model, X_bg, i, j, grid_points=grid_points)

    # 1D PDPs obtained by averaging the 2D surface along axes
    pdp_i = pdp_ij.mean(axis=0)  # average over j (rows)
    pdp_j = pdp_ij.mean(axis=1)  # average over i (cols)

    # Broadcast to full grid
    diff = pdp_ij - (pdp_i[None, :] + pdp_j[:, None])
    num = float(np.sum(diff ** 2))
    den = float(np.sum(pdp_ij ** 2))

    if den <= 0.0:
        return 0.0
    h2 = max(num / den, 0.0)
    return float(np.sqrt(h2))


# ─────────────────────────────────────────────────────────────
# PLOTTING — ICE
# ─────────────────────────────────────────────────────────────
def plot_ice_evolution(models: Dict[int, MLP],
                       X_bg: np.ndarray,
                       feature_idx: int,
                       feature_name: str,
                       annotation_text: str,
                       out_filename: str) -> None:
    """
    Clustered ICE evolution for a single feature across epochs.
    Layout: 2 rows x 3 columns (epochs): row 1 = clustered ICE,
    row 2 = slope histograms.
    """
    epochs = sorted(models.keys())
    grid = np.linspace(0.0, 1.0, GRID_POINTS_1D)

    fig, axes = plt.subplots(
        2, len(epochs), figsize=(15, 8), sharex="col", sharey=False
    )

    # Figure-level title depends on feature (modifier vs crossover)
    if "x0" in feature_name:
        suptitle = (
            "ICE Plot — x0 (Modifier): Clustered by Slope Direction\n"
            "Modifier signature: positive and neutral clusters dominate, negative rare"
        )
    elif "x2" in feature_name:
        suptitle = (
            "ICE Plot — x2 (Crossover): Clustered by Slope Direction\n"
            "Crossover signature: both positive AND negative slope clusters present"
        )
    else:
        suptitle = (
            f"ICE Plot — {feature_name}: Clustered by Slope Direction"
        )

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.99)

    # Cluster colours
    cluster_colors = {
        "pos": "#e74c3c",   # red
        "neu": "#95a5a6",   # grey
        "neg": "#2980b9",   # blue
    }

    for idx, epoch in enumerate(epochs):
        model = models[epoch]
        ice = compute_ice(model, X_bg, feature_idx, grid)  # [n_samples, len(grid)]
        pdp = ice.mean(axis=0)

        # Cluster curves by slope
        sample_pos, sample_neu, sample_neg, slopes = cluster_ice_by_slope(
            ice, grid, n_sample=30, seed=42
        )
        threshold = 0.3 * slopes.std()
        # Full cluster sizes for legend labels
        n_pos = int((slopes > threshold).sum())
        n_neu = int((np.abs(slopes) <= threshold).sum())
        n_neg = int((slopes < -threshold).sum())

        # Row 1: clustered ICE
        ax_ice = axes[0, idx]
        for cluster_name, idx_array, color, label_tpl in [
            ("pos", sample_pos, cluster_colors["pos"],
             f"↗ Positive slope (n={n_pos})"),
            ("neu", sample_neu, cluster_colors["neu"],
             f"→ Neutral slope (n={n_neu})"),
            ("neg", sample_neg, cluster_colors["neg"],
             f"↘ Negative slope (n={n_neg})"),
        ]:
            if len(idx_array) == 0:
                continue
            # Thin sampled curves
            for k in idx_array:
                ax_ice.plot(
                    grid,
                    ice[k],
                    color=color,
                    alpha=0.35,
                    linewidth=0.8,
                )
            # Cluster mean
            cluster_mean = ice[idx_array].mean(axis=0)
            ax_ice.plot(
                grid,
                cluster_mean,
                color=color,
                linewidth=2.5,
                label=label_tpl,
            )

        # Overall PDP (mean over all ICE curves)
        ax_ice.plot(
            grid,
            pdp,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="PDP (overall mean)",
        )

        ax_ice.set_title(f"Epoch {epoch}")
        ax_ice.set_xlabel(feature_name)
        if idx == 0:
            ax_ice.set_ylabel("Model output")
        ax_ice.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

        # Add annotation on the last ICE subplot (epoch 300)
        if idx == len(epochs) - 1 and annotation_text:
            ax_ice.annotate(
                annotation_text,
                xy=(0.97, 0.05),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="#666666",
                    alpha=0.9,
                ),
            )

        # Row 2: histogram of slopes
        ax_hist = axes[1, idx]
        ax_hist.hist(
            slopes,
            bins=30,
            color="#34495e",
            alpha=0.7,
            edgecolor="white",
        )
        ax_hist.axvline(
            threshold,
            color=cluster_colors["pos"],
            linestyle="--",
            linewidth=1.5,
        )
        ax_hist.axvline(
            -threshold,
            color=cluster_colors["neg"],
            linestyle="--",
            linewidth=1.5,
        )
        ax_hist.set_xlabel("ICE curve slope")
        if idx == 0:
            ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Slope distribution — Epoch {epoch}")

    # Shared legend for ICE row
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
    )

    fig.tight_layout(rect=[0.02, 0.05, 1.0, 0.95])
    out_path = os.path.join(BASE_OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ice_centered(model: MLP,
                      X_bg: np.ndarray,
                      feature_idx: int,
                      feature_label: str,
                      out_filename: str) -> None:
    """
    Centered ICE (c-ICE) plot for a single feature at one epoch.
    Each curve is shifted such that its value at grid[0] is zero.
    """
    grid = np.linspace(0.0, 1.0, GRID_POINTS_1D)
    ice = compute_ice(model, X_bg, feature_idx, grid)  # [n_samples, len(grid)]

    # Center each curve
    ice_centered = ice - ice[:, [0]]
    mean_centered = ice_centered.mean(axis=0)
    std_centered = ice_centered.std(axis=0)

    # Randomly sample a subset of curves for visual clarity
    rng = np.random.RandomState(42)
    n_samples = ice_centered.shape[0]
    n_show = min(50, n_samples)
    sample_idx = rng.choice(n_samples, size=n_show, replace=False)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.suptitle(
        f"Centered ICE (c-ICE) — {feature_label}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    # Individual centered ICE curves (sampled)
    for k in sample_idx:
        ax.plot(
            grid,
            ice_centered[k],
            color="#2ca02c",
            alpha=0.3,
            linewidth=0.9,
        )

    # ±1 std band around mean
    ax.fill_between(
        grid,
        mean_centered - std_centered,
        mean_centered + std_centered,
        color="#2ca02c",
        alpha=0.2,
        label="±1 std band",
    )

    # Mean centered curve
    ax.plot(
        grid,
        mean_centered,
        color="#006400",
        linewidth=2.3,
        label="Mean c-ICE (PDP of shape)",
    )

    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature_label)
    ax.set_ylabel("Centered model output")

    ax.annotate(
        "c-ICE removes offset\n→ reveals consistent sin(2πx₄) shape",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f1fff0",
            edgecolor="#228b22",
            alpha=0.9,
        ),
    )
    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(BASE_OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# PLOTTING — H-STATISTIC
# ─────────────────────────────────────────────────────────────
def plot_hstat_evolution(models: Dict[int, MLP],
                         X_bg: np.ndarray,
                         epochs: List[int],
                         out_filename: str) -> None:
    """
    Plot H-statistic over epochs for several feature pairs.
    """
    epochs_sorted = sorted(epochs)

    # Pairs: modifier (true), crossover (true), control (no interaction)
    pairs = [
        (0, 1),  # modifier
        (2, 3),  # crossover
        (0, 2),  # control
    ]
    styles = [
        ("#1f77b4", "-"),   # blue solid
        ("#d62728", "--"),  # red dashed
        ("#7f7f7f", ":"),   # grey dotted
    ]
    labels = [
        "x0×x1 (Modifier — true)",
        "x2×x3 (Crossover — true)",
        "x0×x2 (No interaction — control)",
    ]

    h_values = {pair: [] for pair in pairs}
    for epoch in epochs_sorted:
        model = models[epoch]
        print(f"  Computing H-stat for epoch {epoch}...")
        for pair in pairs:
            h = compute_hstat(model, X_bg, pair[0], pair[1], grid_points=GRID_POINTS_2D)
            h_values[pair].append(h)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig.suptitle(
        "Friedman H-Statistic Evolution — Interaction Strength Over Training",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    for (pair, (color, ls), label) in zip(pairs, styles, labels):
        ax.plot(
            epochs_sorted,
            h_values[pair],
            color=color,
            linestyle=ls,
            marker="o",
            linewidth=2.0,
            label=label,
        )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(min(epochs_sorted), max(epochs_sorted))
    ax.set_ylim(0.0, 1.05 * max(max(vals) for vals in h_values.values()))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("H-statistic (0–1)")
    ax.legend(loc="upper right", fontsize=8)

    ax.annotate(
        "H > 0.1 suggests\nmeaningful interaction",
        xy=(0.02, 0.65),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#fff7e6",
            edgecolor="#e69f00",
            alpha=0.9,
        ),
    )

    fig.tight_layout()
    out_path = os.path.join(BASE_OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_hstat_heatmap(model: MLP,
                       X_bg: np.ndarray,
                       feature_names: List[str],
                       out_filename: str) -> None:
    """
    Heatmap of H-statistic for all pairs among the 5 true features x0..x4.
    Diagonal entries are NaN and shown as white.
    """
    n_true = 5
    H = np.full((n_true, n_true), np.nan, dtype=float)

    # Compute for upper triangle and mirror
    for i in range(n_true):
        for j in range(i + 1, n_true):
            h = compute_hstat(model, X_bg, i, j, grid_points=GRID_POINTS_2D)
            H[i, j] = h
            H[j, i] = h

    # Mask NaNs (diagonal) to show as white
    H_masked = np.ma.masked_invalid(H)
    cmap = plt.get_cmap("Reds")
    cmap = cmap.with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(H_masked, cmap=cmap, vmin=0.0, vmax=np.nanmax(H) * 1.05)

    ax.set_xticks(range(n_true))
    ax.set_yticks(range(n_true))
    ax.set_xticklabels(feature_names[:n_true])
    ax.set_yticklabels(feature_names[:n_true])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.suptitle(
        f"H-Statistic Heatmap at Epoch {EPOCH_HSTAT_300} — Pairwise Interaction Strength",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    # Annotate cells
    for i in range(n_true):
        for j in range(n_true):
            if i == j:
                continue
            val = H[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if val < 0.6 * np.nanmax(H) else "white",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("H-statistic")

    fig.tight_layout()
    out_path = os.path.join(BASE_OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  ICE & H-Statistic Analysis for MLP Snapshots")
    print("=" * 60)

    ensure_dirs()

    print("\nLoading background data...")
    X_bg, feature_names = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    print(f"\nLoading model snapshots for epochs: {EPOCHS_ALL}")
    models = load_unreg_models(EPOCHS_ALL)

    # ICE evolution for x0 (modifier feature)
    print("\n[1/5] ICE evolution for x0 (modifier)...")
    models_x0 = {e: models[e] for e in EPOCHS_ICE}
    plot_ice_evolution(
        models_x0,
        X_bg,
        feature_idx=0,
        feature_name="x0  (modifier feature)",
        annotation_text="Parallel lines → homogeneous modifier effect",
        out_filename="ice_evolution_x0.png",
    )

    # ICE evolution for x2 (crossover feature)
    print("[2/5] ICE evolution for x2 (crossover)...")
    models_x2 = {e: models[e] for e in EPOCHS_ICE}
    plot_ice_evolution(
        models_x2,
        X_bg,
        feature_idx=2,
        feature_name="x2  (crossover feature)",
        annotation_text="Diverging/crossing lines → heterogeneous crossover effect",
        out_filename="ice_evolution_x2.png",
    )

    # Centered ICE for x4 at epoch 300
    print(f"[3/5] Centered ICE for x4 at epoch {EPOCH_HSTAT_300}...")
    model_300 = models[EPOCH_HSTAT_300]
    plot_ice_centered(
        model_300,
        X_bg,
        feature_idx=4,
        feature_label="x4  (nonlinear main effect)",
        out_filename="ice_centered_x4.png",
    )

    # H-statistic evolution across epochs for selected pairs
    print("[4/5] Friedman H-statistic evolution...")
    plot_hstat_evolution(
        models=models,
        X_bg=X_bg,
        epochs=EPOCHS_ALL,
        out_filename="hstat_evolution.png",
    )

    # H-statistic heatmap at epoch 300
    print(f"[5/5] H-statistic heatmap at epoch {EPOCH_HSTAT_300}...")
    plot_hstat_heatmap(
        model=model_300,
        X_bg=X_bg,
        feature_names=feature_names,
        out_filename="hstat_heatmap_epoch300.png",
    )

    print(f"\nDone! All ICE and H-statistic figures saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

