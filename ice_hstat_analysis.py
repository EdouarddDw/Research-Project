"""
ice_hstat_analysis.py
========================

ICE (Individual Conditional Expectation) plots and
Friedman's H-statistic interaction analysis for MLP snapshots.
Static PNG figures only.

Run AFTER train_mlp_snapshots.py has generated unregularized model snapshots in:
    ./outputs/snapshots/unregularized/model_epoch_{N}.pt

All figures are saved under: ./outputs/ice/static/

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
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

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
    "figure.dpi":         220,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SNAPSHOT_DIR       = "./outputs/snapshots"       # unregularized/, l2/ live here
UNREG_SNAPSHOT_DIR = os.path.join(SNAPSHOT_DIR, "unregularized")
ICE_STATIC_DIR     = "./outputs/ice/static"      # all PNG from ice_hstat_analysis.py
ICE_GIFS_DIR       = "./outputs/ice/gifs"       # all GIF from ice_gifs.py

# 9 epochs → 3×3 grid for ALL static plots (ICE, ice_slopes, centered ICE)
_MAX_SNAPSHOT_EPOCH = max(SNAPSHOT_EPOCHS) if SNAPSHOT_EPOCHS else 0
GRID_EPOCHS: List[int] = [
    e for e in [3, 10, 20, 30, 40, 50, 100, 150, 220] if e > 0 and e <= _MAX_SNAPSHOT_EPOCH
]
EPOCHS_ALL: List[int] = [e for e in [1, 10, 50, 100, _MAX_SNAPSHOT_EPOCH] if e > 0]   # for H-stat line plot
EPOCH_HSTAT_300: int = _MAX_SNAPSHOT_EPOCH

GRID_POINTS_1D: int = 60
GRID_POINTS_2D: int = 20
MC_MAX_SAMPLES: int = 1000

GIF_DPI: int = 100
GIF_FPS: int = 4
GIF_INTERVAL_MS: int = 250

# Larger static figures improve readability for 3×3 subplot grids.
STATIC_FIG_DPI = 220
GRID_FIGSIZE_3X3 = (18, 15)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def ensure_dirs() -> None:
    os.makedirs(ICE_STATIC_DIR, exist_ok=True)
    os.makedirs(ICE_GIFS_DIR, exist_ok=True)
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
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
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


def all_feature_pairs(n_features: int) -> List[Tuple[int, int]]:
    """All unique feature pairs (i < j)."""
    return [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]


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
    Centered formula: interaction = PDP_ij - PDP_i - PDP_j + overall_mean.
    """
    xs, ys, pdp_ij = compute_2d_pdp(model, X_bg, i, j, grid_points=grid_points)

    pdp_i = pdp_ij.mean(axis=0)
    pdp_j = pdp_ij.mean(axis=1)
    overall_mean = pdp_ij.mean()

    # Friedman's centered formula
    diff = pdp_ij - pdp_i[None, :] - pdp_j[:, None] + overall_mean
    num = float(np.sum(diff ** 2))
    den = float(np.sum((pdp_ij - overall_mean) ** 2))

    if den <= 0.0:
        return 0.0
    return float(np.sqrt(max(num / den, 0.0)))


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

    # Cluster colours
    cluster_colors = {
        "pos": "#e74c3c",   # red
        "neu": "#95a5a6",   # grey
        "neg": "#2980b9",   # blue
    }

    # PASS 1: precompute everything for global limits
    precomputed = []
    ice_values_all: List[np.ndarray] = []
    slopes_values_all: List[np.ndarray] = []

    for epoch in epochs:
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

        precomputed.append(
            dict(
                epoch=epoch,
                ice=ice,
                pdp=pdp,
                sample_pos=sample_pos,
                sample_neu=sample_neu,
                sample_neg=sample_neg,
                slopes=slopes,
                threshold=threshold,
                n_pos=n_pos,
                n_neu=n_neu,
                n_neg=n_neg,
            )
        )
        ice_values_all.append(ice.ravel())
        slopes_values_all.append(slopes)

    ice_values_all_concat = np.concatenate(ice_values_all)
    global_ymin = float(np.percentile(ice_values_all_concat, 2))
    global_ymax = float(np.percentile(ice_values_all_concat, 98))

    slopes_all_concat = np.concatenate(slopes_values_all)
    global_slope_min = float(slopes_all_concat.min())
    global_slope_max = float(slopes_all_concat.max())
    if global_slope_min == global_slope_max:
        global_slope_min -= 1.0
        global_slope_max += 1.0

    # Fix histogram binning for comparable shapes across epochs
    bins = np.linspace(global_slope_min, global_slope_max, 31)  # 30 bins
    global_count_max = 0
    for item in precomputed:
        counts, _ = np.histogram(item["slopes"], bins=bins)
        if counts.size:
            global_count_max = max(global_count_max, int(counts.max()))
    global_count_max = max(global_count_max, 1)

    # PASS 2: plot with shared y-limits
    fig, axes = plt.subplots(
        2,
        len(epochs),
        figsize=(max(15, 3 * len(epochs)), 3.5 * 2),
        sharex="col",
        sharey=False,
        constrained_layout=True,
    )

    suptitle = f"ICE Plot — {feature_name}: Clustered by Slope Direction"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)

    for idx, item in enumerate(precomputed):
        epoch = item["epoch"]
        ice = item["ice"]
        pdp = item["pdp"]
        slopes = item["slopes"]
        threshold = item["threshold"]

        n_pos = item["n_pos"]
        n_neu = item["n_neu"]
        n_neg = item["n_neg"]

        # Row 1: clustered ICE
        ax_ice = axes[0, idx]
        ax_ice.set_ylim(global_ymin, global_ymax)

        for _, idx_array, color, label_tpl in [
            ("pos", item["sample_pos"], cluster_colors["pos"], f"↗ Positive slope (n={n_pos})"),
            ("neu", item["sample_neu"], cluster_colors["neu"], f"→ Neutral slope (n={n_neu})"),
            ("neg", item["sample_neg"], cluster_colors["neg"], f"↘ Negative slope (n={n_neg})"),
        ]:
            if len(idx_array) == 0:
                continue
            # Light individual curves, bold cluster mean
            for k in idx_array:
                ax_ice.plot(grid, ice[k], color=color, alpha=0.15, linewidth=0.8)
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

        # Row 2: histogram of slopes (shared scale)
        ax_hist = axes[1, idx]
        ax_hist.hist(
            slopes,
            bins=bins,
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
        ax_hist.set_xlim(global_slope_min, global_slope_max)
        ax_hist.set_ylim(0, global_count_max * 1.05)
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

    out_path = os.path.join(ICE_STATIC_DIR, out_filename)
    fig.savefig(out_path, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_ice_individual(models: Dict[int, MLP],
                        X_bg: np.ndarray,
                        feature_names: List[str],
                        feat_indices: List[int]) -> None:
    """
    For each feature in the given index list, save one PNG: 3×3 grid (GRID_EPOCHS).
    Each subplot: clustered ICE curves at that epoch (no histogram row).
    File: ice_evolution_x{i}.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    if not epochs or not feat_indices:
        return
    cluster_colors = {"pos": "#e74c3c", "neu": "#95a5a6", "neg": "#2980b9"}

    for feat_idx in feat_indices:
        feature_name = feature_names[feat_idx]
        print(f"    ICE individual x{feat_idx} ({feature_name})...", flush=True)
        v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
        grid = np.linspace(v_min, v_max, GRID_POINTS_1D)

        # PASS 1: precompute all epochs + global y-limits
        epoch_precomputed = []
        ice_values_all: List[np.ndarray] = []
        for epoch in epochs:
            model = models[epoch]
            ice = compute_ice(model, X_bg, feat_idx, grid)
            pdp = ice.mean(axis=0)
            sample_pos, sample_neu, sample_neg, slopes = cluster_ice_by_slope(
                ice, grid, n_sample=30, seed=42
            )
            threshold = 0.3 * slopes.std()
            n_pos = int((slopes > threshold).sum())
            n_neu = int((np.abs(slopes) <= threshold).sum())
            n_neg = int((slopes < -threshold).sum())
            epoch_precomputed.append(
                dict(
                    epoch=epoch,
                    ice=ice,
                    pdp=pdp,
                    sample_pos=sample_pos,
                    sample_neu=sample_neu,
                    sample_neg=sample_neg,
                    threshold=threshold,
                    n_pos=n_pos,
                    n_neu=n_neu,
                    n_neg=n_neg,
                )
            )
            ice_values_all.append(ice.ravel())

        ice_values_all_concat = np.concatenate(ice_values_all) if ice_values_all else np.array([0.0])
        global_ymin = float(np.percentile(ice_values_all_concat, 2))
        global_ymax = float(np.percentile(ice_values_all_concat, 98))

        # PASS 2: plot with shared y-limits
        fig, axes = plt.subplots(
            3, 3,
            figsize=GRID_FIGSIZE_3X3,
            sharex="col", sharey=False, constrained_layout=True,
        )
        axes_flat = axes.flatten()

        fig.suptitle(
            f"ICE — {feature_name} (x{feat_idx}): Clustered by Slope — Evolution by Epoch",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for idx, item in enumerate(epoch_precomputed):
            epoch = item["epoch"]
            ice = item["ice"]
            pdp = item["pdp"]
            ax_ice = axes_flat[idx]
            ax_ice.set_ylim(global_ymin, global_ymax)

            for _, idx_array, color, label_tpl in [
                ("pos", item["sample_pos"], cluster_colors["pos"], f"↗ Pos (n={item['n_pos']})"),
                ("neu", item["sample_neu"], cluster_colors["neu"], f"→ Neu (n={item['n_neu']})"),
                ("neg", item["sample_neg"], cluster_colors["neg"], f"↘ Neg (n={item['n_neg']})"),
            ]:
                if len(idx_array) == 0:
                    continue
                for k in idx_array:
                    ax_ice.plot(grid, ice[k], color=color, alpha=0.15, linewidth=0.8)
                cluster_mean = ice[idx_array].mean(axis=0)
                ax_ice.plot(grid, cluster_mean, color=color, linewidth=2.5, label=label_tpl)

            ax_ice.plot(grid, pdp, color="black", linestyle="--", linewidth=2.2, label="PDP")
            ax_ice.set_title(f"Epoch {epoch}")
            ax_ice.set_xlabel(feature_name)
            if idx % 3 == 0:
                ax_ice.set_ylabel("Model output")
            ax_ice.xaxis.set_major_locator(ticker.MaxNLocator(6))

        # hide unused subplots (if epochs < 9)
        for idx in range(len(epoch_precomputed), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # Legend from last used subplot
        last_used_idx = len(epoch_precomputed) - 1
        if last_used_idx >= 0:
            handles, labels = axes_flat[last_used_idx].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=4,
                bbox_to_anchor=(0.5, -0.02),
                frameon=True,
            )

        out_path = os.path.join(ICE_STATIC_DIR, f"ice_evolution_x{feat_idx}.png")
        fig.savefig(out_path, dpi=STATIC_FIG_DPI)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_ice_slopes_individual(models: Dict[int, MLP],
                                X_bg: np.ndarray,
                                feature_names: List[str],
                                feat_indices: List[int]) -> None:
    """
    For each feature in the given index list, save one PNG: 3×3 grid of slope histograms.
    File: ice_slopes_x{i}.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    if not epochs or not feat_indices:
        return
    cluster_colors = {"pos": "#e74c3c", "neg": "#2980b9"}

    for feat_idx in feat_indices:
        feature_name = feature_names[feat_idx]
        print(f"    ICE slopes x{feat_idx} ({feature_name})...", flush=True)

        v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
        grid = np.linspace(v_min, v_max, GRID_POINTS_1D)

        # PASS 1: compute slopes for all epochs + global histogram scale
        slopes_per_epoch: List[np.ndarray] = []
        thresholds: List[float] = []
        for epoch in epochs:
            model = models[epoch]
            ice = compute_ice(model, X_bg, feat_idx, grid)
            _, _, _, slopes = cluster_ice_by_slope(ice, grid, n_sample=30, seed=42)
            threshold = 0.3 * slopes.std()
            slopes_per_epoch.append(slopes)
            thresholds.append(float(threshold))

        all_slopes_concat = np.concatenate(slopes_per_epoch) if slopes_per_epoch else np.array([0.0])
        global_slope_min = float(all_slopes_concat.min())
        global_slope_max = float(all_slopes_concat.max())
        if global_slope_min == global_slope_max:
            global_slope_min -= 1.0
            global_slope_max += 1.0

        # Fix histogram bins so shapes are comparable
        bins = np.linspace(global_slope_min, global_slope_max, 31)  # 30 bins

        global_count_max = 0
        for slopes in slopes_per_epoch:
            counts, _ = np.histogram(slopes, bins=bins)
            if counts.size:
                global_count_max = max(global_count_max, int(counts.max()))
        global_count_max = max(global_count_max, 1)

        # PASS 2: plot with shared y scale
        fig, axes = plt.subplots(
            3, 3,
            figsize=GRID_FIGSIZE_3X3,
            sharey=True,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()

        fig.suptitle(
            f"ICE Slope Distribution — x{feat_idx} ({feature_name}) — Evolution by Epoch",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for idx, epoch in enumerate(epochs):
            slopes = slopes_per_epoch[idx]
            threshold = thresholds[idx]
            ax_hist = axes_flat[idx]
            ax_hist.hist(slopes, bins=bins, color="#34495e", alpha=0.7, edgecolor="white")
            ax_hist.axvline(threshold, color=cluster_colors["pos"], linestyle="--", linewidth=1.5)
            ax_hist.axvline(-threshold, color=cluster_colors["neg"], linestyle="--", linewidth=1.5)
            ax_hist.set_xlabel("ICE curve slope")
            ax_hist.set_xlim(global_slope_min, global_slope_max)
            ax_hist.set_ylim(0, global_count_max * 1.05)
            if idx % 3 == 0:
                ax_hist.set_ylabel("Count")
            ax_hist.set_title(f"Epoch {epoch}")

        # hide unused subplots (if epochs < 9)
        for idx in range(len(epochs), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        out_path = os.path.join(ICE_STATIC_DIR, f"ice_slopes_x{feat_idx}.png")
        fig.savefig(out_path, dpi=STATIC_FIG_DPI)
        plt.close(fig)
        print(f"  Saved: {out_path}")


def plot_ice_centered_individual(models: Dict[int, MLP],
                                 X_bg: np.ndarray,
                                 feature_names: List[str],
                                 feat_indices: List[int]) -> None:
    """
    For each feature in the given index list, save one PNG: 3×3 grid (GRID_EPOCHS).
    Each subplot: centered ICE (mean c-ICE ± std) for that epoch.
    File: ice_centered_x{i}.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    if not epochs or not feat_indices:
        return

    for feat_idx in feat_indices:
        feature_name = feature_names[feat_idx]
        print(f"    ICE centered individual x{feat_idx} ({feature_name})...", flush=True)
        v_min, v_max = X_bg[:, feat_idx].min(), X_bg[:, feat_idx].max()
        grid = np.linspace(v_min, v_max, GRID_POINTS_1D)

        # PASS 1: precompute all epochs + global y-limits (include mean±std band)
        epoch_precomputed = []
        all_centered_vals: List[np.ndarray] = []
        for epoch in epochs:
            model = models[epoch]
            ice = compute_ice(model, X_bg, feat_idx, grid)
            ice_centered = ice - ice[:, [0]]
            mean_centered = ice_centered.mean(axis=0)
            std_centered = ice_centered.std(axis=0)
            epoch_precomputed.append(
                dict(
                    epoch=epoch,
                    mean_centered=mean_centered,
                    std_centered=std_centered,
                )
            )
            all_centered_vals.append(ice_centered.ravel())
            all_centered_vals.append((mean_centered - std_centered).ravel())
            all_centered_vals.append((mean_centered + std_centered).ravel())

        all_centered_concat = np.concatenate(all_centered_vals) if all_centered_vals else np.array([0.0])
        global_ymin = float(np.percentile(all_centered_concat, 2))
        global_ymax = float(np.percentile(all_centered_concat, 98))

        # PASS 2: plot with shared y-limits
        fig, axes = plt.subplots(
            3, 3,
            figsize=GRID_FIGSIZE_3X3,
            constrained_layout=True,
        )
        axes_flat = axes.flatten()
        fig.suptitle(
            f"Centered ICE (c-ICE) — x{feat_idx} ({feature_name}) — Evolution by Epoch",
            fontsize=13, fontweight="bold", y=1.02,
        )

        for idx, item in enumerate(epoch_precomputed):
            epoch = item["epoch"]
            mean_centered = item["mean_centered"]
            std_centered = item["std_centered"]
            ax = axes_flat[idx]
            ax.set_ylim(global_ymin, global_ymax)
            ax.fill_between(
                grid,
                mean_centered - std_centered,
                mean_centered + std_centered,
                color="#2ca02c",
                alpha=0.2,
            )
            ax.plot(grid, mean_centered, color="#006400", linewidth=2.3)
            ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
            ax.set_title(f"Epoch {epoch}")
            ax.set_xlabel(feature_name)
            if idx % 3 == 0:
                ax.set_ylabel("Centered model output")

        for idx in range(len(epoch_precomputed), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        out_path = os.path.join(ICE_STATIC_DIR, f"ice_centered_x{feat_idx}.png")
        fig.savefig(out_path, dpi=STATIC_FIG_DPI)
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
    out_path = os.path.join(ICE_STATIC_DIR, out_filename)
    fig.savefig(out_path, dpi=STATIC_FIG_DPI)
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
        ("#2ca02c", "-."),
        ("#9467bd", ":"),
        ("#8c564b", "-"),
        ("#e377c2", "--"),
        ("#7f7f7f", "-."),
        ("#bcbd22", ":"),
        ("#17becf", "-"),
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

    out_path = os.path.join(ICE_STATIC_DIR, out_filename)
    fig.savefig(out_path, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_hstat_heatmap(model: MLP,
                       X_bg: np.ndarray,
                       feature_names: List[str],
                       idx_subset: List[int],
                       out_filename: str) -> None:
    """
    Heatmap of H-statistic for all pairs among idx_subset.
    Diagonal entries are NaN and shown as white.
    """
    n_true = len(idx_subset)
    if n_true == 0:
        print("  [ice_hstat] Empty idx_subset; skipping H-stat heatmap.")
        return
    H = np.full((n_true, n_true), np.nan, dtype=float)

    # Compute for upper triangle and mirror (index into true_idx)
    for ii in range(n_true):
        for jj in range(ii + 1, n_true):
            i, j = idx_subset[ii], idx_subset[jj]
            h = compute_hstat(model, X_bg, i, j, grid_points=GRID_POINTS_2D)
            H[ii, jj] = h
            H[jj, ii] = h

    # Mask NaNs (diagonal) to show as white
    H_masked = np.ma.masked_invalid(H)
    cmap = plt.get_cmap("Reds")
    cmap = cmap.with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(H_masked, cmap=cmap, vmin=0.0, vmax=np.nanmax(H) * 1.05)

    names_subset = [feature_names[k] for k in idx_subset]
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
    out_path = os.path.join(ICE_STATIC_DIR, out_filename)
    fig.savefig(out_path, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_hstat_heatmap_grid(models: Dict[int, MLP],
                             X_bg: np.ndarray,
                             feature_names: List[str],
                             idx_subset: List[int],
                             out_filename: str) -> None:
    """
    3×3 grid of H-stat heatmaps: one subplot per epoch (GRID_EPOCHS).
    Each subplot: heatmap of all pairs among idx_subset at that epoch.
    Shared vmax across all 9 for comparable colour scale.
    File: hstat_heatmap_grid.png
    """
    epochs = [e for e in GRID_EPOCHS if e in models]
    n_true = len(idx_subset)
    if not epochs or n_true == 0:
        print("  [ice_hstat] No GRID_EPOCHS in models or empty idx_subset; skipping H-stat heatmap grid.")
        return

    # Precompute H matrices and global vmax
    H_list = []
    for epoch in epochs:
        model = models[epoch]
        H = np.full((n_true, n_true), np.nan, dtype=float)
        for ii in range(n_true):
            for jj in range(ii + 1, n_true):
                i, j = idx_subset[ii], idx_subset[jj]
                h = compute_hstat(model, X_bg, i, j, grid_points=GRID_POINTS_2D)
                H[ii, jj] = h
                H[jj, ii] = h
        H_list.append(H)
    vmax = max(np.nanmax(H) for H in H_list)
    if not (vmax > 0):
        vmax = 1.0
    vmax = vmax * 1.05

    names_subset = [feature_names[k] for k in idx_subset]
    cmap = plt.get_cmap("Reds").with_extremes(bad="white")

    fig, axes = plt.subplots(
        3, 3,
        figsize=GRID_FIGSIZE_3X3,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    fig.suptitle(
        "H-Statistic Heatmap — Evolution by Epoch",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for idx, (epoch, H) in enumerate(zip(epochs, H_list)):
        ax = axes_flat[idx]
        H_masked = np.ma.masked_invalid(H)
        im = ax.imshow(H_masked, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(n_true))
        ax.set_yticks(range(n_true))
        ax.set_xticklabels(names_subset)
        ax.set_yticklabels(names_subset)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f"Epoch {epoch}")
        for ii in range(n_true):
            for jj in range(n_true):
                if ii == jj:
                    continue
                val = H[ii, jj]
                if np.isnan(val):
                    continue
                ax.text(
                    jj, ii, f"{val:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if val < 0.6 * vmax else "white",
                )

    # Single shared colorbar
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="H-statistic")

    out_path = os.path.join(ICE_STATIC_DIR, out_filename)
    fig.savefig(out_path, dpi=STATIC_FIG_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_hstat_evolution_individual(models: Dict[int, MLP],
                                   X_bg: np.ndarray,
                                   epochs: List[int],
                                   hstat_pairs: List[Tuple[int, int]],
                                   feature_names: List[str]) -> None:
    """
    For each feature pair (i, j), save one PNG: H-stat over epochs (single line).
    File: hstat_evolution_x{i}_x{j}.png
    """
    epochs_sorted = sorted(epochs)
    if not epochs_sorted or not hstat_pairs:
        return

    for pair_idx, (i, j) in enumerate(hstat_pairs):
        print(f"    H-stat evolution {pair_idx + 1}/{len(hstat_pairs)} (x{i}, x{j})...", flush=True)
        vals = []
        for epoch in epochs_sorted:
            model = models[epoch]
            h = compute_hstat(model, X_bg, i, j, grid_points=GRID_POINTS_2D)
            vals.append(h)

        fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
        fig.suptitle(
            f"Friedman H-Statistic — Evolution by Epoch (x{i}×x{j})",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        ax.plot(epochs_sorted, vals, color="#d62728", marker="o", linewidth=2.0)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlim(min(epochs_sorted), max(epochs_sorted))
        ax.set_ylim(0.0, 1.05 * max(vals) if max(vals) > 0 else 1.0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("H-statistic (0–1)")
        ax.set_title(f"Pair: {feature_names[i]} × {feature_names[j]}", fontsize=10, fontweight="bold")

        out_path = os.path.join(ICE_STATIC_DIR, f"hstat_evolution_x{i}_x{j}.png")
        fig.savefig(out_path, dpi=STATIC_FIG_DPI)
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
    all_feature_idx = list(range(len(feature_names)))  # all 10 features for ICE
    all_pairs = all_feature_pairs(len(feature_names))

    print("\nLoading background data...")
    X_bg, _ = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    # Load all snapshot epochs (GRID_EPOCHS, EPOCHS_ALL, etc.)
    epochs_all = sorted(e for e in SNAPSHOT_EPOCHS if e <= 300)
    epochs_for_plots = sorted(set(epochs_all) | set(EPOCHS_ALL) | set(GRID_EPOCHS))
    print(f"\nLoading model snapshots for {len(epochs_for_plots)} epochs...")
    models = load_unreg_models(epochs_for_plots)

    # 1. ICE individual: one PNG per feature — all 10 (3×3 grid)
    if all_feature_idx:
        print("\n[1/6] ICE individual (all features, 3×3 grid)...")
        plot_ice_individual(models, X_bg, feature_names, all_feature_idx)

    # 2. ICE slope histograms: one PNG per feature — all 10 (3×3 grid)
    if all_feature_idx:
        print("[2/6] ICE slopes (3×3 grid)...")
        plot_ice_slopes_individual(models, X_bg, feature_names, all_feature_idx)

    # 3. Centered ICE individual: one PNG per feature — all 10 (3×3 grid)
    if all_feature_idx:
        print("[3/6] Centered ICE individual (all features, 3×3 grid)...")
        plot_ice_centered_individual(models, X_bg, feature_names, all_feature_idx)

    # 4. H-statistic evolution: ALL feature pairs (individual files; readable at 45 pairs)
    hstat_pairs = all_pairs
    pair_labels = [f"x{p[0]}×x{p[1]}" for p in hstat_pairs]
    epochs_hstat = [e for e in GRID_EPOCHS if e in models]
    if epochs_hstat and hstat_pairs:
        print(f"[4/6] Friedman H-statistic evolution (all pairs: {len(hstat_pairs)} figures)...")
        plot_hstat_evolution_individual(
            models={e: models[e] for e in epochs_hstat},
            X_bg=X_bg,
            epochs=epochs_hstat,
            hstat_pairs=hstat_pairs,
            feature_names=feature_names,
        )

    # 5. H-statistic heatmap grid: 3×3 by GRID_EPOCHS (ALL features)
    idx_subset = list(range(len(feature_names)))
    if idx_subset:
        print("[5/6] H-statistic heatmap grid (3×3 by epoch, all features)...")
        plot_hstat_heatmap_grid(
            models=models,
            X_bg=X_bg,
            feature_names=feature_names,
            idx_subset=idx_subset,
            out_filename="hstat_heatmap_grid.png",
        )

    # 6. H-statistic heatmap (single epoch) at the latest snapshot epoch (ALL features)
    if idx_subset and EPOCH_HSTAT_300 in models:
        print(f"[6/6] H-statistic heatmap at epoch {EPOCH_HSTAT_300} (all features)...")
        model_300 = models[EPOCH_HSTAT_300]
        plot_hstat_heatmap(
            model=model_300,
            X_bg=X_bg,
            feature_names=feature_names,
            idx_subset=idx_subset,
            out_filename="hstat_heatmap_epoch300.png",
        )

    print(f"\nDone! All ICE and H-statistic figures saved to: {ICE_STATIC_DIR}")


if __name__ == "__main__":
    main()

