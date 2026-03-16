import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def detect_overfitting_epoch(epochs, train_losses, val_losses, patience=3, min_delta=0.0, smooth_window=3):
    """
    Detect the onset of overfitting more robustly than using the maximum
    validation-train gap.

    Logic:
    1. Smooth train/validation losses with a moving average.
    2. Track the best validation loss seen so far.
    3. Mark overfitting onset as the first epoch after the validation minimum
       where validation loss fails to improve for `patience` consecutive points
       while the train loss is still improving or staying flat.

    Returns
    -------
    overfit_epoch : int
        Estimated epoch where overfitting starts.
    best_epoch : int
        Epoch of best smoothed validation loss.
    smooth_train : np.ndarray
        Smoothed train loss.
    smooth_val : np.ndarray
        Smoothed validation loss.
    """
    epochs = np.asarray(epochs)
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)

    if len(epochs) == 0:
        raise ValueError("epochs must not be empty")

    if not (len(epochs) == len(train_losses) == len(val_losses)):
        raise ValueError("epochs, train_losses, and val_losses must have the same length")

    def moving_average(x, window):
        if window <= 1 or len(x) < window:
            return x.copy()
        kernel = np.ones(window) / window
        smoothed = np.convolve(x, kernel, mode="valid")
        pad_left = window // 2
        pad_right = len(x) - len(smoothed) - pad_left
        return np.pad(smoothed, (pad_left, pad_right), mode="edge")

    smooth_train = moving_average(train_losses, smooth_window)
    smooth_val = moving_average(val_losses, smooth_window)

    best_idx = int(np.argmin(smooth_val))
    best_val = smooth_val[best_idx]
    best_epoch = int(epochs[best_idx])

    stale = 0
    onset_idx = best_idx

    for i in range(best_idx + 1, len(epochs)):
        val_improved = smooth_val[i] < (best_val - min_delta)
        train_not_worse = smooth_train[i] <= (smooth_train[i - 1] + min_delta)

        if val_improved:
            best_val = smooth_val[i]
            best_idx = i
            stale = 0
            continue

        if train_not_worse:
            stale += 1
        else:
            stale = 0

        if stale >= patience:
            onset_idx = i - patience + 1
            break
    else:
        onset_idx = best_idx

    return int(epochs[onset_idx]), int(epochs[best_idx]), smooth_train, smooth_val

import importlib
import multilayer_perceptron
import neural_interaction_detection
import utils


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
SNAPSHOT_EPOCHS = list(range(5, NEPOCHS + 1, 5))  # 1-indexed epochs only
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
        data_loaders = preprocess_data(
            X, Y,
            valid_size=10000,
            test_size=10000,
            std_scale=False,              # <- no normalisation
            get_torch_loaders=True
        )
        print(f"Data: X={X.shape}, Y={Y.shape}")
        print(f"GT interactions: {gt}")
        # 2. Train
        model = MLP(NUM_FEATURES, HIDDEN_UNITS,
                    use_main_effect_nets=USE_MAIN_EFFECT_NETS).to(device)

        model, test_loss, snapshots, history = train(
            model,
            data_loaders,
            device=device,
            nepochs=NEPOCHS,
            learning_rate=LEARNING_RATE,
            verbose=True,
            save_snapshots=True,
            snapshot_epochs=SNAPSHOT_EPOCHS,
            snapshot_dir=snap_dir,
        )
        print(f"Test loss: {test_loss:.4f} | Snapshots: {sorted(snapshots.keys())}")

        epochs_list = history["epoch"]
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]

        # 3. Overfitting detection
        print("Computing train/val losses for overfitting detection...")
        overfit_epoch, best_val_epoch, smooth_train_losses, smooth_val_losses = detect_overfitting_epoch(
            epochs_list,
            train_losses,
            val_losses,
            patience=3,
            min_delta=1e-4,
            smooth_window=3,
        )

        # Plot overfitting
        fig_ov, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs_list, train_losses, label="Train loss",
                color="steelblue", linewidth=1.2, alpha=0.35, marker="o", markersize=3)
        ax.plot(epochs_list, val_losses, label="Validation loss",
                color="darkorange", linewidth=1.2, alpha=0.35, linestyle="--", marker="s", markersize=3)
        ax.plot(epochs_list, smooth_train_losses, label="Train loss (smoothed)",
                color="steelblue", linewidth=2.2)
        ax.plot(epochs_list, smooth_val_losses, label="Validation loss (smoothed)",
                color="darkorange", linewidth=2.2, linestyle="--")
        ax.axvline(best_val_epoch, color="green", linestyle="-.", linewidth=2,
                   label=f"Best val (~epoch {best_val_epoch})")
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
        print(f"  Best val (smoothed): epoch {best_val_epoch}")
        print(f"  Onset:              epoch {overfit_epoch}")

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
