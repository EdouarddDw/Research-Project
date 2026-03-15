"""
ice_gifs.py
===========

GIF animations for ICE and H-statistic analysis: ICE evolution per feature,
H-statistic heatmap evolution. Run when GIFs are needed; ice_hstat_analysis.py
produces only static PNGs.

Imports helpers and constants from ice_hstat_analysis.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ice_hstat_analysis import (
    compute_ice,
    compute_hstat,
    model_predict,
    load_unreg_models,
    load_meta,
    get_background_data,
    ensure_dirs,
    BASE_OUTPUT_DIR,
    UNREG_SNAPSHOT_DIR,
    GRID_POINTS_1D,
    GRID_POINTS_2D,
    MC_MAX_SAMPLES,
    GIF_DPI,
    GIF_FPS,
    GIF_INTERVAL_MS,
)
from train_mlp_snapshots import SNAPSHOT_EPOCHS

# Match ICE/H-stat style for GIF frames
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def create_ice_gif(
    models,
    X_bg,
    feature_idx,
    feature_name,
    all_epochs,
    out_path,
    fps=4,
    n_curve_sample=50,
):
    """One frame per epoch: ICE curves (sampled) + PDP mean. 8×4 in, 100 DPI."""
    epochs = sorted(e for e in all_epochs if e in models)
    if not epochs:
        return
    v_min, v_max = X_bg[:, feature_idx].min(), X_bg[:, feature_idx].max()
    grid = np.linspace(v_min, v_max, GRID_POINTS_1D)
    rng = np.random.RandomState(42)
    n_bg = len(X_bg)
    curve_idx = rng.choice(n_bg, size=min(n_curve_sample, n_bg), replace=False)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    def update(frame_idx):
        ax.clear()
        epoch = epochs[frame_idx]
        model = models[epoch]
        ice = compute_ice(model, X_bg, feature_idx, grid)
        pdp = ice.mean(axis=0)
        for k in curve_idx:
            ax.plot(grid, ice[k], color="#2ca02c", alpha=0.15, linewidth=0.8)
        ax.plot(grid, pdp, color="black", linestyle="--", linewidth=2.2, label="PDP")
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel("Model output", fontsize=10)
        ax.set_title(f"Epoch {epoch}", fontsize=13, fontweight="bold")
        ax.set_xlim(v_min, v_max)
        ax.grid(True, color="#e0e0e0", linestyle="--")
        return []

    ani = FuncAnimation(
        fig, update, frames=len(epochs), interval=GIF_INTERVAL_MS, blit=False
    )
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def create_hstat_heatmap_gif(
    models,
    X_bg,
    feature_names,
    true_idx,
    all_epochs,
    out_path,
    fps=4,
):
    """One frame per epoch: H-stat heatmap over signal features. Global vmax."""
    epochs = sorted(e for e in all_epochs if e in models)
    n_true = len(true_idx)
    if not epochs or n_true == 0:
        return
    # Precompute H matrices and global vmax
    H_list = []
    for epoch in epochs:
        model = models[epoch]
        H = np.full((n_true, n_true), np.nan, dtype=float)
        for ii in range(n_true):
            for jj in range(ii + 1, n_true):
                i, j = true_idx[ii], true_idx[jj]
                h = compute_hstat(model, X_bg, i, j, grid_points=GRID_POINTS_2D)
                H[ii, jj] = H[jj, ii] = h
        H_list.append(H)
    vmax = max(np.nanmax(H) for H in H_list) * 1.05
    names_subset = [feature_names[k] for k in true_idx]
    cmap = plt.get_cmap("Reds").with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)

    def update(frame_idx):
        ax.clear()
        H = H_list[frame_idx]
        H_masked = np.ma.masked_invalid(H)
        im = ax.imshow(H_masked, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_xticks(range(n_true))
        ax.set_yticks(range(n_true))
        ax.set_xticklabels(names_subset)
        ax.set_yticklabels(names_subset)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for ii in range(n_true):
            for jj in range(n_true):
                if ii == jj:
                    continue
                val = H[ii, jj]
                if np.isnan(val):
                    continue
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if val < 0.6 * vmax else "white")
        ax.set_title(f"Epoch {epochs[frame_idx]}", fontsize=13, fontweight="bold")
        return []

    ani = FuncAnimation(
        fig, update, frames=len(epochs), interval=GIF_INTERVAL_MS, blit=False
    )
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    # Load meta, background, models (all epochs)
    print("=" * 60)
    print("  ICE & H-Stat GIFs — ICE Evolution & H-Stat Heatmap Animations")
    print("=" * 60)

    ensure_dirs()

    print("\nLoading meta (feature_names, ground_truth)...")
    feature_names, ground_truth = load_meta()
    true_idx = ground_truth["true_idx"]

    print("\nLoading background data...")
    X_bg, _ = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    epochs_all = sorted(e for e in SNAPSHOT_EPOCHS if e <= 300)
    print(f"\nLoading model snapshots for {len(epochs_all)} epochs...")
    models = load_unreg_models(epochs_all)

    # create_ice_gif per signal feature
    for fid in true_idx:
        out_gif = os.path.join(BASE_OUTPUT_DIR, f"ice_evolution_x{fid}.gif")
        print(f"[1/2] ICE GIF for {feature_names[fid]}...")
        create_ice_gif(
            models, X_bg, fid, feature_names[fid], epochs_all, out_gif, fps=GIF_FPS
        )

    # create_hstat_heatmap_gif
    if true_idx:
        out_hstat_gif = os.path.join(BASE_OUTPUT_DIR, "hstat_heatmap_evolution.gif")
        print("[2/2] H-statistic heatmap GIF...")
        create_hstat_heatmap_gif(
            models, X_bg, feature_names, true_idx, epochs_all, out_hstat_gif, fps=GIF_FPS
        )

    print(f"\nDone! All GIFs saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
