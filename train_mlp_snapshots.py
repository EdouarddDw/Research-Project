"""
train_mlp_snapshots.py
======================
Group Project: The Evolution of Logic in AI
-------------------------------------------
This module handles:
  1. Synthetic data from synth.py (e.g. F3) with known ground-truth interactions
  2. MLP training via multilayer_perceptron.MLP + train() with epoch-level snapshots
  3. Saving snapshots for downstream PDP / NID analysis
  4. Uses utils.preprocess_data() for splitting and optional scaling.

Usage:
    python train_mlp_snapshots.py

Outputs (saved to ./outputs/snapshots/):
    - unregularized/model_epoch_{N}.pt  — model weights at each snapshot epoch
    - unregularized/loss_curves.csv      — train/val loss per epoch (recomputed from snapshots)
    - unregularized/meta.pkl             — feature names, ground truth, synth_fn_idx
    - l2/model_epoch_{N}.pt, loss_curves.csv, meta.pkl (L2-regularized run)
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synth
from multilayer_perceptron import MLP, train, get_weights
from utils import preprocess_data

# ─────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SYNTH_FN_IDX   = 2             # 0-indexed: 0=F1, 2=F3, etc.
NOISE_STD      = 0.0           # noise for synth functions
NUM_FEATURES   = 10            # all synth functions have 10 features
HIDDEN_UNITS   = [140, 100, 60, 20]
USE_MAIN_EFFECT_NETS = True
NUM_SAMPLES    = 3000

EPOCHS         = 300
LR             = 1e-2           # multilayer_perceptron default is 1e-2
L1_CONST       = 5e-5          # L1 on interaction_mlp weights
L2_CONST       = 0.0

# Dense early (1–30), sparse late (40–300) for maturation vs saturation/decay
SNAPSHOT_EPOCHS = sorted(set(
    list(range(1, 31)) + [40, 50, 60, 75, 90, 100, 125, 150, 175, 200, 250, 300]
))

BASE_OUTPUT_DIR = "./outputs/snapshots"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  DATA FROM SYNTH
# ─────────────────────────────────────────────
def generate_data_from_synth(fn_idx, n_samples=3000, seed=42, noise_std=0.0):
    """
    Call synth.functions[fn_idx] and return data in a standard format.

    Returns:
        dict with keys: X (float32), y (float32), ground_truth, feature_names

    IMPORTANT: synth.py returns GT sets as 1-indexed (e.g., {1,2} = features x1, x2).
    This function converts to 0-indexed for the pipeline.
    """
    fn = synth.functions[fn_idx]
    X, y, gt_sets = fn(num_samples=n_samples, seed=seed, noise_std=noise_std)

    n_features = X.shape[1]
    feature_names = [f"x{i}" for i in range(n_features)]

    pairwise = []
    any_order = []
    signal_idx = set()

    for s in gt_sets:
        s_0idx = sorted(i - 1 for i in s)
        any_order.append(tuple(s_0idx))
        signal_idx.update(s_0idx)
        for a in range(len(s_0idx)):
            for b in range(a + 1, len(s_0idx)):
                pair = (s_0idx[a], s_0idx[b])
                if pair not in pairwise:
                    pairwise.append(pair)

    ground_truth = {
        "pairwise":  pairwise,
        "any_order": any_order,
        "true_idx":  sorted(signal_idx),
        "noise_idx": sorted(set(range(n_features)) - signal_idx),
    }

    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.float32),
        "ground_truth": ground_truth,
        "feature_names": feature_names,
    }


# ─────────────────────────────────────────────
# 2.  LOSS RECORDS FROM SNAPSHOTS (Option B: no change to multilayer_perceptron)
# ─────────────────────────────────────────────
def recompute_losses_from_snapshots(snapshots, data_loaders, device=None):
    """
    Recompute train and val MSE for each snapshot (like interaction_animation_all_functions).
    Returns list of dicts: [{"epoch": int, "train_loss": float, "val_loss": float}, ...]
    """
    if device is None:
        device = torch.device("cpu")
    criterion = nn.MSELoss(reduction="mean")
    loss_records = []
    for epoch, state_dict in sorted(snapshots.items()):
        tmp = MLP(NUM_FEATURES, HIDDEN_UNITS,
                  use_main_effect_nets=USE_MAIN_EFFECT_NETS).to(device)
        tmp.load_state_dict(state_dict)
        tmp.eval()
        with torch.no_grad():
            t_losses = []
            for Xb, Yb in data_loaders["train"]:
                Xb, Yb = Xb.to(device), Yb.to(device)
                pred = tmp(Xb)
                if pred.dim() == 2:
                    pred = pred.squeeze(-1)
                if Yb.dim() == 2:
                    Yb = Yb.squeeze(-1)
                t_losses.append(criterion(pred, Yb).item())
            v_losses = []
            for Xb, Yb in data_loaders["val"]:
                Xb, Yb = Xb.to(device), Yb.to(device)
                pred = tmp(Xb)
                if pred.dim() == 2:
                    pred = pred.squeeze(-1)
                if Yb.dim() == 2:
                    Yb = Yb.squeeze(-1)
                v_losses.append(criterion(pred, Yb).item())
        loss_records.append({
            "epoch": epoch,
            "train_loss": np.mean(t_losses),
            "val_loss": np.mean(v_losses),
        })
    return loss_records


# ─────────────────────────────────────────────
# 3.  PLOTTING HELPERS (kept for backward compatibility)
# ─────────────────────────────────────────────
def plot_loss_curves(loss_df: pd.DataFrame,
                     save_path: str = None,
                     output_dir: str = None):
    if output_dir is None:
        output_dir = BASE_OUTPUT_DIR

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
    ax.plot(loss_df["epoch"], loss_df["val_loss"],   label="Val Loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Curves — Train vs Validation")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    path = save_path or os.path.join(output_dir, "loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def find_overfitting_epoch(loss_df: pd.DataFrame) -> int | None:
    """
    Heuristic: first epoch where val_loss increases relative to the best
    value seen so far (indicating onset of overfitting).
    """
    best_val = float("inf")
    overfit_epoch = None
    for _, row in loss_df.sort_values("epoch").iterrows():
        epoch = int(row["epoch"])
        val = float(row["val_loss"])
        if val < best_val - 1e-8:
            best_val = val
        elif val > best_val * 1.001:
            overfit_epoch = epoch
            break
    return overfit_epoch


def plot_comparison_loss_curves(unreg_loss_df: pd.DataFrame,
                                l2_loss_df: pd.DataFrame,
                                save_path: str | None = None):
    """
    Plot train/val loss for both unregularized and L2-regularized models
    on the same axes (4 curves total).
    """
    if save_path is None:
        save_path = os.path.join(
            BASE_OUTPUT_DIR, "comparison_loss_curves.png"
        )

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        unreg_loss_df["epoch"],
        unreg_loss_df["train_loss"],
        color="red",
        linestyle="-",
        label="Unregularized Train",
    )
    ax.plot(
        unreg_loss_df["epoch"],
        unreg_loss_df["val_loss"],
        color="red",
        linestyle="--",
        label="Unregularized Val",
    )
    ax.plot(
        l2_loss_df["epoch"],
        l2_loss_df["train_loss"],
        color="blue",
        linestyle="-",
        label="L2 Train",
    )
    ax.plot(
        l2_loss_df["epoch"],
        l2_loss_df["val_loss"],
        color="blue",
        linestyle="--",
        label="L2 Val",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Curves — Unregularized vs L2-Regularized")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# 4.  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Evolution of Logic in AI — MLP Training + Snapshots")
    print("=" * 60)
    print(f"Training on synth.functions[{SYNTH_FN_IDX}] (F{SYNTH_FN_IDX + 1})")

    # 1. Generate data
    data = generate_data_from_synth(SYNTH_FN_IDX, NUM_SAMPLES, SEED, NOISE_STD)
    X, y = data["X"], data["y"]
    feature_names = data["feature_names"]
    ground_truth  = data["ground_truth"]

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  True features : {ground_truth['true_idx']}")
    print(f"  Noise features: {ground_truth['noise_idx']}")
    print(f"  Pairwise GT   : {ground_truth['pairwise']}\n")

    # 2. Preprocess using supervisor's utility
    data_loaders = preprocess_data(
        X, y,
        valid_size=500,
        test_size=500,
        std_scale=False,  # keep raw feature space so PDP/ICE background matches
        get_torch_loaders=True,
        batch_size=256,
    )

    unreg_dir = os.path.join(BASE_OUTPUT_DIR, "unregularized")
    l2_dir    = os.path.join(BASE_OUTPUT_DIR, "l2")

    def run_experiment(name, l1_const, l2_const, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        model = MLP(NUM_FEATURES, HIDDEN_UNITS,
                    use_main_effect_nets=USE_MAIN_EFFECT_NETS)

        model, test_loss, snapshots = train(
            model, data_loaders,
            nepochs=EPOCHS,
            learning_rate=LR,
            l1_const=l1_const,
            l2_const=l2_const,
            verbose=True,
            early_stopping=False,
            save_snapshots=True,
            snapshot_epochs=SNAPSHOT_EPOCHS,
            snapshot_dir=output_dir,
        )

        # Recompute train/val losses from snapshots (Option B)
        loss_records = recompute_losses_from_snapshots(snapshots, data_loaders)
        loss_df = pd.DataFrame(loss_records)
        loss_df.to_csv(os.path.join(output_dir, "loss_curves.csv"), index=False)

        meta = {
            "feature_names": feature_names,
            "ground_truth": ground_truth,
            "synth_fn_idx": SYNTH_FN_IDX,
            "noise_std": NOISE_STD,
        }
        with open(os.path.join(output_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        print(f"\n{name} complete. Test loss: {test_loss:.5f}")
        return model, test_loss, snapshots, loss_df

    unreg_model, unreg_loss, _, unreg_loss_df = run_experiment(
        "Unregularized", l1_const=0.0, l2_const=0.0, output_dir=unreg_dir
    )
    l2_model, l2_loss, _, l2_loss_df = run_experiment(
        "L2 Regularized", l1_const=0.0, l2_const=1e-4, output_dir=l2_dir
    )

    print("\nGenerating diagnostic plots...")
    plot_loss_curves(unreg_loss_df, output_dir=unreg_dir)
    plot_loss_curves(l2_loss_df, output_dir=l2_dir)
    plot_comparison_loss_curves(
        unreg_loss_df,
        l2_loss_df,
        save_path=os.path.join(BASE_OUTPUT_DIR, "comparison_loss_curves.png"),
    )

    print("\nDone! Key files for PDP module:")
    print(f"  {unreg_dir}/model_epoch_*.pt  — unregularized model snapshots")
    print(f"  {l2_dir}/model_epoch_*.pt    — L2-regularized model snapshots")
    print(f"  {unreg_dir}/meta.pkl          — feature names + ground truth")


if __name__ == "__main__":
    main()
