import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from multilayer_perceptron import MLP, train, get_weights
from neural_interaction_detection import get_interactions, interactions_to_matrix
from utils import preprocess_data, set_seed, detect_overfitting_epoch
import synth


# ── Exact notebook model/training config ─────────────────────────────────────
NUM_FEATURES = 10
NUM_SAMPLES = 1000
HIDDEN_UNITS = [256, 256]
USE_MAIN_EFFECT_NETS = True

NEPOCHS = 250
LEARNING_RATE = 1e-2

SAVE_SNAPSHOTS = True
SNAPSHOT_EPOCHS = [i for i in range(0, NEPOCHS + 1, 5)]
LABELS = [f"x_{i}" for i in range(1, NUM_FEATURES + 1)]

VALID_SIZE = 400
TEST_SIZE = 200
STD_SCALE = False

NOISE_LEVELS = [0.0, 0.2, 0.5]
FUNCTION_INDICES = list(range(len(synth.functions)))

BASE_OUT = "./outputs/all_functions_update"


def get_gt_pairwise(gt):
    """Robust extraction of GT pairwise interactions from dict/list formats."""
    if isinstance(gt, dict):
        gt_pairwise = gt.get("pairwise", [])
    else:
        gt_pairwise = [inter for inter in gt if len(inter) == 2]

    out = []
    for pair in gt_pairwise:
        if len(pair) != 2:
            continue
        i1, j1 = sorted([int(x) for x in pair])
        out.append((i1, j1))
    return out


def compute_pairwise_matrices_from_snapshots(snapshots):
    pairwise_matrices = {}
    for epoch, state_dict in sorted(snapshots.items()):
        temp_model = MLP(NUM_FEATURES, HIDDEN_UNITS, use_main_effect_nets=USE_MAIN_EFFECT_NETS)
        temp_model.load_state_dict(state_dict)
        temp_model.eval()

        weights = get_weights(temp_model)
        pairwise = get_interactions(weights, pairwise=True, one_indexed=True)
        pairwise_matrices[epoch] = interactions_to_matrix(pairwise, NUM_FEATURES)

    available_epochs = sorted(pairwise_matrices.keys())
    return pairwise_matrices, available_epochs


def plot_overfitting(history, out_path, title):
    epochs_list = history["epoch"]
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    overfit_epoch, best_val_epoch, smooth_train, smooth_val = detect_overfitting_epoch(
        epochs=epochs_list,
        train_losses=train_losses,
        val_losses=val_losses,
        patience=5,
        min_delta=1e-4,
        smooth_window=7,
    )

    fig_ov, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_list, train_losses, label="Train loss (raw)",
            color="steelblue", linewidth=1.5, alpha=0.35)
    ax.plot(epochs_list, val_losses, label="Validation loss (raw)",
            color="darkorange", linewidth=1.5, alpha=0.35)
    ax.plot(epochs_list, smooth_train, label="Train loss (smoothed)",
            color="steelblue", linewidth=2.2)
    ax.plot(epochs_list, smooth_val, label="Validation loss (smoothed)",
            color="darkorange", linewidth=2.2, linestyle="--")

    ax.axvline(best_val_epoch, color="green", linestyle="-.", linewidth=2,
               label=f"Best val epoch ({best_val_epoch})")
    ax.axvline(overfit_epoch, color="red", linestyle=":", linewidth=2,
               label=f"Overfitting onset (~epoch {overfit_epoch})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(epochs_list[0], epochs_list[-1])

    plt.tight_layout()
    fig_ov.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_ov)


def save_pairwise_gif(pairwise_matrices, available_epochs, gt_pairwise, out_path, vmax_pw):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pairwise_matrices[available_epochs[0]], cmap="viridis", vmin=0, vmax=vmax_pw)

    ax.set_xticks(np.arange(NUM_FEATURES))
    ax.set_yticks(np.arange(NUM_FEATURES))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title("Pairwise Interactions")
    fig.colorbar(im, ax=ax, label="Strength", fraction=0.046, pad=0.04)
    title = fig.suptitle(f"Epoch {available_epochs[0]}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # GT red squares (both symmetric cells)
    for i1, j1 in gt_pairwise:
        i0, j0 = i1 - 1, j1 - 1
        ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=2.0, zorder=5))
        ax.add_patch(Rectangle((i0 - 0.5, j0 - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=2.0, zorder=5))

    annot = [[ax.text(j, i, "", ha="center", va="center", fontsize=7)
              for j in range(NUM_FEATURES)] for i in range(NUM_FEATURES)]

    def update(frame_idx):
        epoch = available_epochs[frame_idx]
        pw_mat = pairwise_matrices[epoch]
        im.set_data(pw_mat)
        title.set_text(f"Epoch {epoch}")

        for i in range(NUM_FEATURES):
            for j in range(NUM_FEATURES):
                val = pw_mat[i, j]
                annot[i][j].set_text(f"{val:.1f}" if val > 0.05 else "")
                annot[i][j].set_color("white" if val > vmax_pw * 0.5 else "black")

        return [im, title] + [a for row in annot for a in row]

    ani = FuncAnimation(fig, update, frames=len(available_epochs), interval=600, blit=False)
    ani.save(out_path, writer="pillow", fps=2)
    plt.close(fig)


def save_pairwise_grid(pairwise_matrices, available_epochs, gt_pairwise, out_path, vmax_pw):
    n_epochs = len(available_epochs)
    fig3, axes3 = plt.subplots(1, n_epochs, figsize=(2.5 * n_epochs, 3.2), constrained_layout=True)

    if n_epochs == 1:
        axes3 = [axes3]

    for col, epoch in enumerate(available_epochs):
        ax_pw = axes3[col]
        ax_pw.imshow(pairwise_matrices[epoch], cmap="viridis", vmin=0, vmax=vmax_pw)
        ax_pw.set_title(f"Epoch {epoch}", fontsize=9)
        ax_pw.set_xticks([])
        ax_pw.set_yticks([])

        for i1, j1 in gt_pairwise:
            i0, j0 = i1 - 1, j1 - 1
            ax_pw.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                                      edgecolor="red", linewidth=1.8, zorder=5))
            ax_pw.add_patch(Rectangle((i0 - 0.5, j0 - 0.5), 1, 1, fill=False,
                                      edgecolor="red", linewidth=1.8, zorder=5))

    axes3[0].set_ylabel("Pairwise", fontsize=10)
    fig3.suptitle("Pairwise Interaction Strength Across Training Epochs", fontsize=12, fontweight="bold")

    fig3.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)


def main():
    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(BASE_OUT, exist_ok=True)

    for fn_idx in FUNCTION_INDICES:
        for noise_std in NOISE_LEVELS:
            print("\n" + "=" * 72)
            print(f"Function F{fn_idx} | noise_std={noise_std}")
            print("=" * 72)

            run_dir = os.path.join(BASE_OUT, f"F{fn_idx}", f"noise_{noise_std}")
            snapshot_dir = os.path.join(run_dir, "snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)

            gif_path = os.path.join(run_dir, "interaction_evolution.gif")
            grid_path = os.path.join(run_dir, "interaction_evolution_grid.png")
            overfit_path = os.path.join(run_dir, "overfitting_detection.png")

            # Data generation (same style as notebook)
            X, Y, gt = synth.functions[fn_idx](
                num_samples=NUM_SAMPLES,
                seed=42,
                noise_std=noise_std,
            )
            data_loaders = preprocess_data(
                X,
                Y,
                valid_size=VALID_SIZE,
                test_size=TEST_SIZE,
                std_scale=STD_SCALE,
                get_torch_loaders=True,
            )

            print(f"Data shape: X={X.shape}, Y={Y.shape}")
            if isinstance(gt, dict):
                print(f"Ground truth pairwise interactions: {gt.get('pairwise', [])}")
                print(f"Ground truth any-order interactions: {gt.get('any_order', [])}")
            else:
                print(f"Ground truth interactions: {gt}")

            # Train (same as notebook)
            model = MLP(NUM_FEATURES, HIDDEN_UNITS, use_main_effect_nets=USE_MAIN_EFFECT_NETS).to(device)
            result = train(
                model,
                data_loaders,
                device=device,
                nepochs=NEPOCHS,
                learning_rate=LEARNING_RATE,
                verbose=True,
                save_snapshots=SAVE_SNAPSHOTS,
                snapshot_epochs=SNAPSHOT_EPOCHS,
                snapshot_dir=snapshot_dir,
                sanity_check_every=10,
                use_adam=True,
                weight_decay=0.0,
            )

            model, test_loss, snapshots, history = result
            print(f"Final test loss: {test_loss:.4f}")

            # Overfitting plot (same style)
            plot_overfitting(
                history,
                overfit_path,
                title="Overfitting Detection: Train vs Validation Loss",
            )
            print(f"Overfitting plot saved to: {overfit_path}")

            # Pairwise interaction matrices from snapshots
            pairwise_matrices, available_epochs = compute_pairwise_matrices_from_snapshots(snapshots)
            if not available_epochs:
                print("No snapshots available, skipping interaction visuals.")
                continue

            all_pairwise = np.array([pairwise_matrices[e] for e in available_epochs])
            vmax_pw = all_pairwise.max()
            print(f"Pairwise range: 0 – {vmax_pw:.2f}")

            gt_pairwise = get_gt_pairwise(gt)

            # GIF + static grid
            save_pairwise_gif(pairwise_matrices, available_epochs, gt_pairwise, gif_path, vmax_pw)
            print(f"Animation saved to: {gif_path}")

            save_pairwise_grid(pairwise_matrices, available_epochs, gt_pairwise, grid_path, vmax_pw)
            print(f"Static grid saved to: {grid_path}")

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
