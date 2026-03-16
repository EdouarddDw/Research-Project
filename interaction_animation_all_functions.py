import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from multilayer_perceptron import MLP, train
from utils import (
    preprocess_data,
    create_interaction_animation,
    set_seed,
    detect_overfitting_epoch,
    compute_interaction_matrices_from_snapshots,
)
import synth
from nid_plots import NIDPlotConfig, build_non_gt_dataframe, plot_non_gt_emergence

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

# ── Output directories ─────────────────────────────────────────────────────────
BASE_OUT = "./outputs/all_functions"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and visualize interaction evolution across synthetic functions."
    )
    parser.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Run a quick smoke test with reduced data/epochs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print run configuration without training or writing outputs.",
    )
    return parser.parse_args()


def main(test_mode: bool = False, dry_run: bool = False):
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    if test_mode:
        print("Running QUICK TEST mode (-test): reduced data, epochs, functions, and noise levels.")
        run_num_samples = 3000
        run_nepochs = 10
        run_snapshot_epochs = [5, 10]
        run_functions = [0]
        run_noise_levels = [0.1]
        run_valid_size = 500
        run_test_size = 500
        run_base_out = "./outputs/all_functions_test"
    else:
        run_num_samples = NUM_SAMPLES
        run_nepochs = NEPOCHS
        run_snapshot_epochs = SNAPSHOT_EPOCHS
        run_functions = FUNCTIONS
        run_noise_levels = NOISE_LEVELS
        run_valid_size = 10000
        run_test_size = 10000
        run_base_out = BASE_OUT

    if dry_run:
        print("DRY RUN (no training will be executed)")
        print(f"  device: {device}")
        print(f"  base_out: {os.path.abspath(run_base_out)}")
        print(f"  functions: {run_functions}")
        print(f"  noise_levels: {run_noise_levels}")
        print(f"  num_samples: {run_num_samples}")
        print(f"  epochs: {run_nepochs}")
        print(f"  snapshot_epochs: {run_snapshot_epochs}")
        print(f"  valid_size/test_size: {run_valid_size}/{run_test_size}")
        return

    os.makedirs(run_base_out, exist_ok=True)

    # ── Main loop ──────────────────────────────────────────────────────────────
    for noise_std in run_noise_levels:
        for fn_idx in run_functions:
            print(f"\n{'='*60}")
            print(f"  Function F{fn_idx} | noise_std = {noise_std}")
            print(f"{'='*60}")

            # Per-function, per-noise output paths
            fn_dir       = os.path.join(run_base_out, f"F{fn_idx}", f"noise_{noise_std}")
            snap_dir     = os.path.join(fn_dir, "snapshots")
            gif_path     = os.path.join(fn_dir, "interaction_evolution.gif")
            grid_path    = os.path.join(fn_dir, "interaction_evolution_grid.png")
            overfit_path = os.path.join(fn_dir, "overfitting_detection.png")
            os.makedirs(snap_dir, exist_ok=True)

            # 1. Generate data
            X, Y, gt = synth.functions[fn_idx](num_samples=run_num_samples, seed=42, noise_std=noise_std)
            data_loaders = preprocess_data(
                X,
                Y,
                valid_size=run_valid_size,
                test_size=run_test_size,
                std_scale=False,
                get_torch_loaders=True,
            )
            print(f"Data: X={X.shape}, Y={Y.shape}")
            print(f"GT interactions: {gt}")
            # 2. Train
            model = MLP(NUM_FEATURES, HIDDEN_UNITS,
                        use_main_effect_nets=USE_MAIN_EFFECT_NETS).to(device)

            model, test_loss, _snapshots, history = train(
                model,
                data_loaders,
                device=device,
                nepochs=run_nepochs,
                learning_rate=LEARNING_RATE,
                verbose=True,
                save_snapshots=True,
                snapshot_epochs=run_snapshot_epochs,
                snapshot_dir=snap_dir,
            )
            print(f"Test loss: {test_loss:.4f}")

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
            pairwise_matrices, anyorder_matrices, available_epochs = compute_interaction_matrices_from_snapshots(
                snapshot_dir=snap_dir,
                snapshot_epochs=run_snapshot_epochs,
                num_features=NUM_FEATURES,
                hidden_units=HIDDEN_UNITS,
                use_main_effect_nets=USE_MAIN_EFFECT_NETS,
            )
            if not available_epochs:
                print(f"No snapshots found in {snap_dir}; skipping interaction visuals.")
                continue

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

    # ── 8. Build NID emergence plot across all generated runs ─────────────────
    print("\nGenerating NID non-GT emergence plot...")

    nid_config = NIDPlotConfig(
        base_out=os.path.abspath(run_base_out),
        noise_levels=tuple(run_noise_levels),
        num_features=NUM_FEATURES,
        hidden_units=tuple(HIDDEN_UNITS),
        use_main_effect_nets=USE_MAIN_EFFECT_NETS,
    )

    nid_df = build_non_gt_dataframe(config=nid_config)
    if nid_df.empty:
        print("NID plot skipped: no snapshot data found for configured runs.")
        return

    nid_csv_path = os.path.join(os.path.abspath(run_base_out), "non_gt_emergence_metrics.csv")
    nid_png_path = os.path.join(os.path.abspath(run_base_out), "non_gt_emergence_noise_comparison.png")

    nid_df.to_csv(nid_csv_path, index=False)
    print(f"Saved: {nid_csv_path}")

    plot_non_gt_emergence(
        nid_df,
        save_path=nid_png_path,
        title="Emergence of non-ground-truth interactions during training",
    )

    print(f"Saved: {nid_png_path}")


if __name__ == "__main__":
    args = parse_args()
    main(test_mode=args.test, dry_run=args.dry_run)
