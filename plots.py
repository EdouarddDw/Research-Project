import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_mlp_snapshots import BASE_OUTPUT_DIR

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

    # Unregularized
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

    # L2-regularized
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
