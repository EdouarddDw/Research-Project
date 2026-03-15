"""
ice_hstat_analysis.py
========================

ICE (Individual Conditional Expectation) plots and
Friedman's H-statistic interaction analysis for MLP snapshots.
Static PNG figures only.

Run AFTER train_mlp_snapshots.py has generated unregularized model snapshots in:
    ./outputs/snapshots/unregularized/model_epoch_{N}.pt

All figures are saved under: ./outputs/snapshots/

For GIF animations run separately: python ice_gifs.py
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap

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
# GLOBAL STYLE — typography, layout (match PDP)
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


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_OUTPUT_DIR = "./outputs/snapshots"
UNREG_SNAPSHOT_DIR = os.path.join(BASE_OUTPUT_DIR, "unregularized")

EPOCHS_ALL: List[int] = [1, 10, 50, 100, 300]   # for H-stat line plot
EPOCHS_ICE: List[int] = [1, 50, 300]            # for static ICE grid
EPOCH_HSTAT_300: int = 300

GRID_POINTS_1D: int = 60
GRID_POINTS_2D: int = 20
MC_MAX_SAMPLES: int = 1000

GIF_DPI: int = 100
GIF_FPS: int = 4
GIF_INTERVAL_MS: int = 250


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


def load_meta() -> Tuple[List[str], dict]:
    """Load feature_names and ground_truth from meta.pkl saved by train_mlp_snapshots."""
    meta_path = os.path.join(UNREG_SNAPSHOT_DIR, "meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["feature_names"], meta["ground_truth"]


def load_unreg_models(epochs: List[int]) -> Dict[int, MLP]:
    """
    Load unregularized models for the specified epochs.
    """
    models: Dict[int, MLP] = {}
    for epoch in epochs:
        path = os.path.join(UNREG_SNAPSHOT_DIR, f"model_epoch_{epoch}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing snapshot: {path}")
        model = MLP(NUM_FEATURES, HIDDEN_UNITS, use_main_effect_nets=USE_MAIN_EFFECT_NETS)
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
    data = generate_data_from_synth(SYNTH_FN_IDX, NUM_SAMPLES, SEED, NOISE_STD)
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
    multilayer_perceptron.MLP returns (batch, 1); squeeze to (batch,).
    """
    with torch.no_grad():
        x_t = torch.from_numpy(X.astype(np.float32))
        preds = model(x_t).squeeze(-1).cpu().numpy()
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
    Grid uses data range for each feature.
    Returns xs, ys, Z with shape [grid_points] x [grid_points].
    """
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
    row 2 = slope histograms. Grid uses data range for the feature.
    """
    epochs = sorted(models.keys())
    v_min, v_max = X_bg[:, feature_idx].min(), X_bg[:, feature_idx].max()
    grid = np.linspace(v_min, v_max, GRID_POINTS_1D)

    fig, axes = plt.subplots(
        2, len(epochs), figsize=(max(15, 3 * len(epochs)), 3.5 * 2),
        sharex="col", sharey=False, constrained_layout=True
    )

    suptitle = f"ICE Plot — {feature_name}: Clustered by Slope Direction"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

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
            # Light individual curves, bold cluster mean
            for k in idx_array:
                ax_ice.plot(
                    grid,
                    ice[k],
                    color=color,
                    alpha=0.15,
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

        # Overall PDP (mean over all ICE curves) — bold black
        ax_ice.plot(
            grid,
            pdp,
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="PDP (overall mean)",
        )

        ax_ice.set_title(f"Epoch {epoch}")
        ax_ice.set_xlabel(feature_name)
        if idx == 0:
            ax_ice.set_ylabel("Model output")
        ax_ice.xaxis.set_major_locator(ticker.MaxNLocator(6))

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
    Grid uses data range for the feature.
    """
    v_min, v_max = X_bg[:, feature_idx].min(), X_bg[:, feature_idx].max()
    grid = np.linspace(v_min, v_max, GRID_POINTS_1D)
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

    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    fig.suptitle(
        f"Centered ICE (c-ICE) — {feature_label}",
        fontsize=13,
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
        "c-ICE removes level offset\n→ reveals pure feature effect shape",
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
                         hstat_pairs: List[Tuple[int, int]],
                         feature_names: List[str],
                         pair_labels: List[str],
                         out_filename: str) -> None:
    """
    Plot H-statistic over epochs for given feature pairs.
    hstat_pairs: list of (i, j); pair_labels: one label per pair.
    """
    epochs_sorted = sorted(epochs)
    pairs = hstat_pairs
    styles = [
        ("#1f77b4", "-"),
        ("#d62728", "--"),
        ("#7f7f7f", ":"),
        ("#2ca02c", "-."),
    ]
    labels = pair_labels[: len(pairs)]
    while len(styles) < len(pairs):
        styles.append(("#7f7f7f", ":"))

    h_values = {pair: [] for pair in pairs}
    for epoch in epochs_sorted:
        model = models[epoch]
        print(f"  Computing H-stat for epoch {epoch}...")
        for pair in pairs:
            h = compute_hstat(model, X_bg, pair[0], pair[1], grid_points=GRID_POINTS_2D)
            h_values[pair].append(h)

    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    fig.suptitle(
        "Friedman H-Statistic Evolution — Interaction Strength Over Training",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for idx, pair in enumerate(pairs):
        color, ls = styles[idx % len(styles)]
        label = labels[idx] if idx < len(labels) else f"x{pair[0]}×x{pair[1]}"
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

    out_path = os.path.join(BASE_OUTPUT_DIR, out_filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_hstat_heatmap(model: MLP,
                       X_bg: np.ndarray,
                       feature_names: List[str],
                       true_idx: List[int],
                       out_filename: str) -> None:
    """
    Heatmap of H-statistic for all pairs among signal features (true_idx).
    Diagonal entries are NaN and shown as white.
    """
    n_true = len(true_idx)
    if n_true == 0:
        print("  [ice_hstat] No true_idx; skipping H-stat heatmap.")
        return
    H = np.full((n_true, n_true), np.nan, dtype=float)

    # Compute for upper triangle and mirror (index into true_idx)
    for ii in range(n_true):
        for jj in range(ii + 1, n_true):
            i, j = true_idx[ii], true_idx[jj]
            h = compute_hstat(model, X_bg, i, j, grid_points=GRID_POINTS_2D)
            H[ii, jj] = h
            H[jj, ii] = h

    # Mask NaNs (diagonal) to show as white
    H_masked = np.ma.masked_invalid(H)
    cmap = plt.get_cmap("Reds")
    cmap = cmap.with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(H_masked, cmap=cmap, vmin=0.0, vmax=np.nanmax(H) * 1.05)

    names_subset = [feature_names[k] for k in true_idx]
    ax.set_xticks(range(n_true))
    ax.set_yticks(range(n_true))
    ax.set_xticklabels(names_subset)
    ax.set_yticklabels(names_subset)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.suptitle(
        f"H-Statistic Heatmap at Epoch {EPOCH_HSTAT_300} — Pairwise Interaction Strength",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Annotate cells
    for ii in range(n_true):
        for jj in range(n_true):
            if ii == jj:
                continue
            val = H[ii, jj]
            if np.isnan(val):
                continue
            ax.text(
                jj,
                ii,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if val < 0.6 * np.nanmax(H) else "white",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("H-statistic")
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

    print("\nLoading meta (feature_names, ground_truth)...")
    feature_names, ground_truth = load_meta()
    pairs = ground_truth["pairwise"]
    true_idx = ground_truth["true_idx"]
    noise_idx = ground_truth["noise_idx"]

    print("\nLoading background data...")
    X_bg, _ = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    # Load all snapshot epochs for GIFs; also need EPOCHS_ALL and EPOCHS_ICE for static plots
    epochs_all = sorted(e for e in SNAPSHOT_EPOCHS if e <= 300)
    epochs_for_plots = sorted(set(epochs_all) | set(EPOCHS_ALL) | set(EPOCHS_ICE))
    print(f"\nLoading model snapshots for {len(epochs_for_plots)} epochs...")
    models = load_unreg_models(epochs_for_plots)
    models_subset = {e: models[e] for e in EPOCHS_ICE}

    # 1. ICE evolution grid (static) for first two GT pair features
    if len(pairs) >= 1:
        print("\n[1/5] ICE evolution grid (first GT pair feature)...")
        i0 = pairs[0][0]
        plot_ice_evolution(
            models_subset,
            X_bg,
            feature_idx=i0,
            feature_name=feature_names[i0],
            annotation_text="ICE evolution for signal feature",
            out_filename=f"ice_evolution_x{i0}.png",
        )
    if len(pairs) >= 2:
        print("[2/5] ICE evolution grid (second GT pair feature)...")
        i1 = pairs[1][0]
        plot_ice_evolution(
            models_subset,
            X_bg,
            feature_idx=i1,
            feature_name=feature_names[i1],
            annotation_text="ICE evolution for signal feature",
            out_filename=f"ice_evolution_x{i1}.png",
        )

    # 2. Centered ICE at final epoch
    if true_idx and EPOCH_HSTAT_300 in models:
        print(f"[3/5] Centered ICE at epoch {EPOCH_HSTAT_300}...")
        model_300 = models[EPOCH_HSTAT_300]
        fid = true_idx[0]
        plot_ice_centered(
            model_300,
            X_bg,
            feature_idx=fid,
            feature_label=feature_names[fid],
            out_filename=f"ice_centered_x{fid}.png",
        )

    # 3. H-statistic evolution (line plot)
    hstat_pairs = list(pairs[:2])
    pair_labels = [f"x{p[0]}×x{p[1]} (GT)" for p in hstat_pairs]
    if len(noise_idx) >= 2:
        hstat_pairs.append((noise_idx[0], noise_idx[1]))
        pair_labels.append(f"x{noise_idx[0]}×x{noise_idx[1]} (noise)")
    print("[4/5] Friedman H-statistic evolution...")
    plot_hstat_evolution(
        models={e: models[e] for e in EPOCHS_ALL if e in models},
        X_bg=X_bg,
        epochs=EPOCHS_ALL,
        hstat_pairs=hstat_pairs,
        feature_names=feature_names,
        pair_labels=pair_labels,
        out_filename="hstat_evolution.png",
    )

    # 4. H-statistic heatmap (static) at epoch 300
    if true_idx and EPOCH_HSTAT_300 in models:
        print(f"[5/5] H-statistic heatmap at epoch {EPOCH_HSTAT_300}...")
        plot_hstat_heatmap(
            model=model_300,
            X_bg=X_bg,
            feature_names=feature_names,
            true_idx=true_idx,
            out_filename="hstat_heatmap_epoch300.png",
        )

    print(f"\nDone! All ICE and H-statistic figures saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

