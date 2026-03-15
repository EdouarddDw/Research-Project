"""
pdp_gifs.py
===========

GIF animations for PDP analysis: 2D PDP per pair, combined 2D PDP, interaction signature.
Run when GIFs are needed; pdp_analysis.py produces only static PNGs.

Imports helpers and constants from pdp_analysis.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from pdp_analysis import (
    compute_2d_pdp,
    compute_interaction_curves,
    model_predict,
    load_unreg_models,
    load_meta,
    get_background_data,
    ensure_dirs,
    BASE_OUTPUT_DIR,
    UNREG_SNAPSHOT_DIR,
    GRID_EPOCHS_2D,
    INTERACTION_EPOCH,
    GRID_POINTS_1D,
    GRID_POINTS_2D,
    MC_MAX_SAMPLES,
    GIF_DPI,
    GIF_FPS,
    GIF_INTERVAL_MS,
    COND_COLORS,
)
from train_mlp_snapshots import SNAPSHOT_EPOCHS

# Match PDP style for GIF frames
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


def create_pdp_gif(models, X_bg, i, j, feature_names, all_epochs, out_path, fps=4):
    """Animate 2D PDP heatmap across epochs. Colorbar created once (vmin/vmax global)."""
    epochs = sorted(e for e in all_epochs if e in models)
    if not epochs:
        return
    # Precompute
    all_Z = []
    n_epochs = len(epochs)
    for ei, epoch in enumerate(epochs):
        _, _, Z = compute_2d_pdp(models[epoch], X_bg, i, j)
        all_Z.append(Z)
        if (ei + 1) % 10 == 0 or ei == 0 or ei == n_epochs - 1:
            print(f"      frame {ei + 1}/{n_epochs}", flush=True)
    xs, ys, _ = compute_2d_pdp(models[epochs[0]], X_bg, i, j)
    vmin = min(z.min() for z in all_Z)
    vmax = max(z.max() for z in all_Z)
    Xg, Yg = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    # Colorbar ONCE on first frame; vmin/vmax fixed → scale correct for all frames
    cf = ax.contourf(Xg, Yg, all_Z[0], levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cf, ax=ax, label="Avg output")

    def update(frame_idx):
        ax.clear()
        ax.contourf(Xg, Yg, all_Z[frame_idx], levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(Xg, Yg, all_Z[frame_idx], levels=6, colors="white", linewidths=0.4, alpha=0.6)
        ax.set_xlabel(feature_names[i], fontsize=10)
        ax.set_ylabel(feature_names[j], fontsize=10)
        ax.set_title(f"Epoch {epochs[frame_idx]}", fontsize=13, fontweight="bold")
        return []

    ani = FuncAnimation(fig, update, frames=len(epochs), interval=GIF_INTERVAL_MS, blit=False)
    print(f"      writing GIF...", flush=True)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def create_pdp_combined_gif(models, X_bg, pair1, pair2, feature_names, all_epochs, out_path, fps=4):
    """Side-by-side 2D PDP for top 2 GT pairs, one frame per epoch."""
    epochs = sorted(e for e in all_epochs if e in models)
    if not epochs or len(pair1) != 2 or len(pair2) != 2:
        return
    i1, j1 = pair1
    i2, j2 = pair2
    Z1_list, Z2_list = [], []
    n_epochs = len(epochs)
    for ei, epoch in enumerate(epochs):
        _, _, Z1 = compute_2d_pdp(models[epoch], X_bg, i1, j1)
        _, _, Z2 = compute_2d_pdp(models[epoch], X_bg, i2, j2)
        Z1_list.append(Z1)
        Z2_list.append(Z2)
        if (ei + 1) % 10 == 0 or ei == 0 or ei == n_epochs - 1:
            print(f"      combined GIF frame {ei + 1}/{n_epochs}", flush=True)
    xs1, ys1, _ = compute_2d_pdp(models[epochs[0]], X_bg, i1, j1)
    xs2, ys2, _ = compute_2d_pdp(models[epochs[0]], X_bg, i2, j2)
    vmin1, vmax1 = min(z.min() for z in Z1_list), max(z.max() for z in Z1_list)
    vmin2, vmax2 = min(z.min() for z in Z2_list), max(z.max() for z in Z2_list)
    Xg1, Yg1 = np.meshgrid(xs1, ys1)
    Xg2, Yg2 = np.meshgrid(xs2, ys2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        ax1.contourf(Xg1, Yg1, Z1_list[frame_idx], levels=20, cmap="viridis", vmin=vmin1, vmax=vmax1)
        ax1.contour(Xg1, Yg1, Z1_list[frame_idx], levels=6, colors="white", linewidths=0.4, alpha=0.6)
        ax1.set_xlabel(feature_names[i1], fontsize=10)
        ax1.set_ylabel(feature_names[j1], fontsize=10)
        ax1.set_title(f"({feature_names[i1]}, {feature_names[j1]})", fontsize=10, fontweight="bold")
        ax2.contourf(Xg2, Yg2, Z2_list[frame_idx], levels=20, cmap="viridis", vmin=vmin2, vmax=vmax2)
        ax2.contour(Xg2, Yg2, Z2_list[frame_idx], levels=6, colors="white", linewidths=0.4, alpha=0.6)
        ax2.set_xlabel(feature_names[i2], fontsize=10)
        ax2.set_ylabel(feature_names[j2], fontsize=10)
        ax2.set_title(f"({feature_names[i2]}, {feature_names[j2]})", fontsize=10, fontweight="bold")
        fig.suptitle(f"Epoch {epochs[frame_idx]}", fontsize=13, fontweight="bold", y=1.02)
        return []

    ani = FuncAnimation(
        fig, update, frames=len(epochs), interval=GIF_INTERVAL_MS, blit=False
    )
    print(f"      writing combined GIF...", flush=True)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def create_interaction_signature_gif(models, X_bg, pair1, pair2, feature_names, all_epochs, out_path, fps=4):
    """Animate interaction signature (two panels) across training epochs."""
    epochs = sorted(e for e in all_epochs if e in models)
    if not epochs:
        return
    cond_pct = [0.25, 0.5, 0.75]
    cond_values1 = [np.percentile(X_bg[:, pair1[1]], p * 100) for p in cond_pct]
    cond_values2 = [np.percentile(X_bg[:, pair2[1]], p * 100) for p in cond_pct]
    linestyles = ["-", "--", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    n_epochs = len(epochs)
    def update(frame_idx):
        epoch = epochs[frame_idx]
        if frame_idx % 10 == 0 or frame_idx == n_epochs - 1:
            print(f"      interaction sig frame {frame_idx + 1}/{n_epochs}", flush=True)
        model = models[epoch]
        grid1, curves1 = compute_interaction_curves(
            model, X_bg, var_idx=pair1[0], cond_idx=pair1[1], cond_values=cond_values1
        )
        grid2, curves2 = compute_interaction_curves(
            model, X_bg, var_idx=pair2[0], cond_idx=pair2[1], cond_values=cond_values2
        )
        ax1.clear()
        ax2.clear()
        for c, col, ls in zip(cond_values1, COND_COLORS, linestyles):
            ax1.plot(grid1, curves1[c], color=col, linewidth=2.2, linestyle=ls, label=f"{c:.2f}")
        ax1.axhline(0, color="#aaaaaa", linewidth=0.8)
        ax1.set_title(f"({feature_names[pair1[0]]}, {feature_names[pair1[1]]})", fontsize=10, fontweight="bold")
        ax1.set_xlabel(feature_names[pair1[0]], fontsize=10)
        ax1.set_ylabel("Avg. Model Output (PDP)", fontsize=10)
        ax1.legend(title=f"{feature_names[pair1[1]]}", fontsize=8)
        for c, col, ls in zip(cond_values2, COND_COLORS, linestyles):
            ax2.plot(grid2, curves2[c], color=col, linewidth=2.2, linestyle=ls, label=f"{c:.2f}")
        ax2.axhline(0, color="#aaaaaa", linewidth=0.8)
        ax2.set_title(f"({feature_names[pair2[0]]}, {feature_names[pair2[1]]})", fontsize=10, fontweight="bold")
        ax2.set_xlabel(feature_names[pair2[0]], fontsize=10)
        ax2.legend(title=f"{feature_names[pair2[1]]}", fontsize=8)
        fig.suptitle(f"Epoch {epoch}", fontsize=13, fontweight="bold", y=1.02)
        return []

    ani = FuncAnimation(
        fig, update, frames=len(epochs), interval=GIF_INTERVAL_MS, blit=False
    )
    print(f"      writing interaction signature GIF...", flush=True)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=GIF_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    # Load meta, background, models (all epochs)
    print("=" * 60)
    print("  PDP GIFs — 2D PDP & Interaction Signature Animations")
    print("=" * 60)

    ensure_dirs()

    print("\nLoading meta (feature_names, ground_truth)...")
    feature_names, ground_truth = load_meta()
    pairs = ground_truth["pairwise"]

    print("\nLoading background data...")
    X_bg, _ = get_background_data(MC_MAX_SAMPLES)
    print(f"  Background samples: {X_bg.shape}")

    epochs_all = sorted(e for e in SNAPSHOT_EPOCHS if e <= 300)
    print(f"\nLoading model snapshots for {len(epochs_all)} epochs...")
    models = load_unreg_models(epochs_all)

    # Representative pairs (one per any_order GT set, max 4 rows)
    representative_pairs = []
    seen = set()
    for ao in ground_truth["any_order"]:
        ao_sorted = sorted(ao)
        if len(ao_sorted) >= 2:
            p = (ao_sorted[0], ao_sorted[1])
            if p not in seen:
                representative_pairs.append(p)
                seen.add(p)
    if not representative_pairs:
        representative_pairs = list(pairs)

    # create_pdp_gif per representative pair
    for gif_num, (i, j) in enumerate(representative_pairs, 1):
        out_gif = os.path.join(BASE_OUTPUT_DIR, f"pdp_2d_x{i}_x{j}.gif")
        print(f"[1/3] 2D PDP GIF {gif_num}/{len(representative_pairs)} ({feature_names[i]}, {feature_names[j]})...")
        create_pdp_gif(models, X_bg, i, j, feature_names, epochs_all, out_gif, fps=GIF_FPS)

    # create_pdp_combined_gif
    if len(pairs) >= 2:
        out_combined = os.path.join(BASE_OUTPUT_DIR, "pdp_2d_combined.gif")
        print("[2/3] 2D PDP combined GIF...")
        create_pdp_combined_gif(
            models, X_bg, pairs[0], pairs[1], feature_names, epochs_all, out_combined, fps=GIF_FPS
        )

    # create_interaction_signature_gif
    if len(pairs) >= 2:
        out_sig_gif = os.path.join(BASE_OUTPUT_DIR, "pdp_interaction_signature.gif")
        print("[3/3] Interaction signature GIF...")
        create_interaction_signature_gif(
            models, X_bg, pairs[0], pairs[1], feature_names, epochs_all, out_sig_gif, fps=GIF_FPS
        )

    print(f"\nDone! All GIFs saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
