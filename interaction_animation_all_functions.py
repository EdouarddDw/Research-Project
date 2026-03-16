import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import importlib
import multilayer_perceptron
import neural_interaction_detection
import utils
importlib.reload(multilayer_perceptron)
importlib.reload(neural_interaction_detection)
importlib.reload(utils)

from multilayer_perceptron import MLP, train, get_weights
from neural_interaction_detection import get_interactions, interactions_to_matrix
from utils import preprocess_data, load_snapshots, create_interaction_animation
import synth

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_FEATURES   = 10
NUM_SAMPLES    = 30000
HIDDEN_UNITS   = [140, 100, 60, 20]
USE_MAIN_EFFECT_NETS = True
NEPOCHS        = 200
LEARNING_RATE  = 1e-2
L1_CONST       = 5e-5
SNAPSHOT_EPOCHS = list(range(0, 201, 5))
LABELS         = [f"x_{i}" for i in range(1, NUM_FEATURES + 1)]
FUNCTIONS      = list(range(len(synth.functions)))
NOISE_LEVELS   = [0.1, 0.5]

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# ── Output directories ─────────────────────────────────────────────────────────
BASE_OUT = "./outputs/all_functions"
os.makedirs(BASE_OUT, exist_ok=True)

# ── Main loop ──────────────────────────────────────────────────────────────────
for noise_std in NOISE_LEVELS:
    for fn_idx in FUNCTIONS:
        print(f"\n{'='*60}")
        print(f"  Function F{fn_idx} | noise_std = {noise_std}")
        print(f"{'='*60}")

        # Per-function, per-noise output paths
        fn_dir       = os.path.join(BASE_OUT, f"F{fn_idx}", f"noise_{noise_std}")
        snap_dir     = os.path.join(fn_dir, "snapshots")
        gif_path     = os.path.join(fn_dir, "interaction_evolution.gif")
        grid_path    = os.path.join(fn_dir, "interaction_evolution_grid.png")
        overfit_path = os.path.join(fn_dir, "overfitting_detection.png")
        os.makedirs(snap_dir, exist_ok=True)

        # 1. Generate data
        X, Y, gt = synth.functions[fn_idx](num_samples=NUM_SAMPLES, seed=42, noise_std=noise_std)
        data_loaders = preprocess_data(X, Y, valid_size=10000, test_size=10000,
                                       std_scale=True, get_torch_loaders=True)
        print(f"Data: X={X.shape}, Y={Y.shape}")
        print(f"GT interactions: {gt}")
        # 2. Train
        model = MLP(NUM_FEATURES, HIDDEN_UNITS,
                    use_main_effect_nets=USE_MAIN_EFFECT_NETS).to(device)

        model, test_loss, snapshots = train(
            model, data_loaders,
            device=device,
            nepochs=NEPOCHS,
            learning_rate=LEARNING_RATE,
            l1_const=L1_CONST,
            l2_const=0.0,
            verbose=True,
            early_stopping=False,
            save_snapshots=True,
            snapshot_epochs=SNAPSHOT_EPOCHS,
            snapshot_dir=snap_dir,
        )
        print(f"Test loss: {test_loss:.4f} | Snapshots: {sorted(snapshots.keys())}")

        # 3. Overfitting detection
        print("Computing train/val losses for overfitting detection...")
        criterion = torch.nn.MSELoss(reduction="mean")
        train_losses = []
        val_losses   = []
        epochs_list  = sorted(snapshots.keys())

        for epoch, state_dict in sorted(snapshots.items()):
            tmp = MLP(NUM_FEATURES, HIDDEN_UNITS,
                      use_main_effect_nets=USE_MAIN_EFFECT_NETS).to(device)
            tmp.load_state_dict(state_dict)
            tmp.eval()

            with torch.no_grad():
                t_losses = [
                    criterion(tmp(Xb.to(device)).squeeze(), Yb.to(device)).item()
                    for Xb, Yb in data_loaders["train"]
                ]
                v_losses = [
                    criterion(tmp(Xb.to(device)).squeeze(), Yb.to(device)).item()
                    for Xb, Yb in data_loaders["val"]
                ]

            train_losses.append(np.mean(t_losses))
            val_losses.append(np.mean(v_losses))

        # Detect overfitting onset
        gap = np.array(val_losses) - np.array(train_losses)
        if gap.max() > 0:
            onset_idx = int(np.argmax(gap > gap.max() * 0.1))
        else:
            onset_idx = len(epochs_list) - 1
        overfit_epoch = epochs_list[onset_idx]

        # Plot overfitting
        fig_ov, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs_list, train_losses, label="Train loss",
                color="steelblue", linewidth=2, marker="o", markersize=4)
        ax.plot(epochs_list, val_losses, label="Validation loss",
                color="darkorange", linewidth=2, linestyle="--", marker="s", markersize=4)
        ax.axvline(overfit_epoch, color="red", linestyle=":", linewidth=2,
                   label=f"Overfitting onset (~epoch {overfit_epoch})")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title(f"F{fn_idx} (noise={noise_std}) — Overfitting Detection",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(epochs_list[0], epochs_list[-1])
        plt.tight_layout()
        fig_ov.savefig(overfit_path, dpi=150, bbox_inches="tight")
        plt.close(fig_ov)
        
        print(f"Overfitting plot saved: {overfit_path}")
        print(f"  Min train: {min(train_losses):.4f} @ epoch {epochs_list[int(np.argmin(train_losses))]}")
        print(f"  Min val:   {min(val_losses):.4f} @ epoch {epochs_list[int(np.argmin(val_losses))]}")
        print(f"  Onset:     epoch {overfit_epoch}")

        # 4. Compute interactions per snapshot
        pairwise_matrices = {}
        anyorder_matrices = {}

        for epoch, state_dict in sorted(snapshots.items()):
            tmp = MLP(NUM_FEATURES, HIDDEN_UNITS,
                      use_main_effect_nets=USE_MAIN_EFFECT_NETS)
            tmp.load_state_dict(state_dict)
            tmp.eval()

            weights  = get_weights(tmp)
            pairwise = get_interactions(weights, pairwise=True,  one_indexed=True)
            anyorder = get_interactions(weights, pairwise=False, one_indexed=True)

            pairwise_matrices[epoch] = interactions_to_matrix(pairwise, NUM_FEATURES)
            anyorder_matrices[epoch] = interactions_to_matrix(anyorder, NUM_FEATURES)

        available_epochs = sorted(pairwise_matrices.keys())

        # 5. Colour scale
        all_pw = np.array([pairwise_matrices[e] for e in available_epochs])
        all_ao = np.array([anyorder_matrices[e] for e in available_epochs])
        vmax_pw = all_pw.max()
        vmax_ao = all_ao.max()
        print(f"Pairwise range: 0–{vmax_pw:.2f} | Any-order range: 0–{vmax_ao:.2f}")

        # 6. Save GIF
        fig, update = create_interaction_animation(
            pairwise_matrices, anyorder_matrices, available_epochs,
            LABELS, vmax_pw, vmax_ao
        )
        ani = FuncAnimation(fig, update, frames=len(available_epochs),
                            interval=600, blit=False)
        ani.save(gif_path, writer="pillow", fps=2)
        plt.close(fig)
        print(f"GIF saved: {gif_path}")

        # 7. Save static grid
        n = len(available_epochs)
        fig3, axes3 = plt.subplots(2, n, figsize=(2.5 * n, 6), constrained_layout=True)

        for col, epoch in enumerate(available_epochs):
            axes3[0, col].imshow(pairwise_matrices[epoch], cmap="viridis",
                                 vmin=0, vmax=vmax_pw)
            axes3[0, col].set_title(f"Ep {epoch}", fontsize=8)
            axes3[0, col].axis("off")

            axes3[1, col].imshow(anyorder_matrices[epoch], cmap="magma",
                                 vmin=0, vmax=vmax_ao)
            axes3[1, col].axis("off")

        axes3[0, 0].set_ylabel("Pairwise",  fontsize=10)
        axes3[1, 0].set_ylabel("Any-Order", fontsize=10)
        fig3.suptitle(f"F{fn_idx} (noise={noise_std}) — Interaction Evolution",
                      fontsize=12, fontweight="bold")
        fig3.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Grid saved: {grid_path}")

print("\n" + "="*60)
print("All functions and noise levels complete.")
print("="*60)
