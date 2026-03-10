import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import shap
import matplotlib.pyplot as plt

def plot_loss_curves(loss_df: pd.DataFrame,
                     save_path: str = None,
                     output_dir: str = "./outputs/snapshots"):
    if output_dir is None:
        output_dir = "./outputs/snapshots"

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

def plot_shap_summary(shap_vals, X_eval, feature_names, save_path, title):
    plt.figure(figsize=(9,6))
    shap.summary_plot(shap_vals, X_eval, feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_shap_bar(importance, feature_names, save_path, title):
    order = np.argsort(importance)[::-1]
    plt.figure(figsize=(8,4))
    plt.bar(np.array(feature_names)[order], importance[order])
    plt.xticks(rotation=45)
    plt.ylabel("mean(|SHAP|)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_shap_importance_vs_epoch(epochs, importance_matrix, feature_names, save_path):
    plt.figure(figsize=(10,6))
    for i, name in enumerate(feature_names):
        plt.plot(
            epochs,
            importance_matrix[:, i],
            marker="o",
            linewidth=1.5,
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("mean(|SHAP|)")
    plt.title("Feature importance vs training epoch")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_noise_vs_true_importance(epochs, importance_matrix, feature_names, save_path):
    # detect true vs noise features from names
    true_idx = [i for i,f in enumerate(feature_names) if "true" in f or f in ["x1","x2","x3","x4","x5"]]
    noise_idx = [i for i,f in enumerate(feature_names) if "noise" in f or f in ["x6","x7","x8","x9","x10"]]

    true_importance = np.mean(importance_matrix[:, true_idx], axis=1)
    noise_importance = np.mean(importance_matrix[:, noise_idx], axis=1)

    plt.figure(figsize=(8,5))

    plt.plot(epochs, true_importance,
             marker="o",
             linewidth=2,
             label="True features")
    plt.plot(epochs, noise_importance,
             marker="o",
             linewidth=2,
             label="Noise features")

    plt.xlabel("Epoch")
    plt.ylabel("Mean |SHAP|")
    plt.title("True vs Noise Feature Importance")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()