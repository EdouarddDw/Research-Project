"""
analyze_shap_snapshots.py
=========================

Compute SHAP explanations for MLP snapshots produced by
train_mlp_snapshots.py.
"""

import os
import re
import numpy as np
import torch
import shap
from pathlib import Path

from train_mlp_snapshots import generate_data, MLP, N_SAMPLES, N_TRUE, N_NOISE, SEED
from plots import plot_shap_summary, plot_shap_bar, plot_shap_importance_vs_epoch, plot_noise_vs_true_importance

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SNAPSHOT_ROOT = "./outputs/snapshots"
OUTPUT_ROOT   = "./outputs/shap_analysis"

BACKGROUND_SIZE = 100
EVAL_SIZE       = 500

HIDDEN = [140, 100, 60, 20]

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ─────────────────────────────────────────────
# MODEL WRAPPER (Fix for SHAP)
# ─────────────────────────────────────────────
class ModelWrapper(torch.nn.Module):
    """
    SHAP requires model outputs to have shape (batch, outputs).
    Our model returns (batch,), so we add an extra dimension.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.unsqueeze(1)

# ─────────────────────────────────────────────
# FIND SNAPSHOTS
# ─────────────────────────────────────────────
def discover_snapshots(folder):
    pattern = re.compile(r"model_epoch_(\d+)\.pt$")
    snapshots = []
    for p in Path(folder).glob("model_epoch_*.pt"):
        m = pattern.match(p.name)
        if m:
            snapshots.append((int(m.group(1)), p))
    snapshots.sort(key=lambda x: x[0])
    return snapshots


# ─────────────────────────────────────────────
# SHAP ANALYSIS FOR ONE RUN
# ─────────────────────────────────────────────
def analyze_run(run_name):
    print(f"\nAnalyzing run: {run_name}")
    snapshot_dir = os.path.join(SNAPSHOT_ROOT, run_name)
    output_dir   = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(output_dir, exist_ok=True)
    snapshots = discover_snapshots(snapshot_dir)
    print(f"Found {len(snapshots)} snapshots")

    # Load dataset (same generation as training)
    data = generate_data(N_SAMPLES, N_TRUE, N_NOISE, seed=SEED)
    X = data["X"]
    feature_names = [f"x{i+1}" for i in range(X.shape[1])]

    # Random subsets for SHAP
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    bg_idx   = idx[:BACKGROUND_SIZE]
    eval_idx = idx[BACKGROUND_SIZE:BACKGROUND_SIZE + EVAL_SIZE]
    X_bg   = torch.tensor(X[bg_idx])
    X_eval = torch.tensor(X[eval_idx])

    epoch_importance = {}
    for epoch, path in snapshots:
        print(f"  Epoch {epoch}")

        # Rebuild model
        model = MLP(X.shape[1], hidden=HIDDEN)
        model.load_state_dict(torch.load(path))
        model.eval()

        # Wrap model so SHAP receives (batch,1) output
        wrapped_model = ModelWrapper(model)

        # SHAP computation
        explainer = shap.GradientExplainer(wrapped_model, X_bg)
        shap_vals = explainer.shap_values(X_eval)
        shap_vals = np.array(shap_vals).squeeze()

        # Global feature importance
        importance = np.mean(np.abs(shap_vals), axis=0)
        epoch_importance[epoch] = importance

        # Save raw SHAP values
        np.save(os.path.join(output_dir, f"shap_epoch_{epoch}.npy"), shap_vals)

    epochs = sorted(epoch_importance.keys())
    importance_matrix = np.vstack([epoch_importance[e] for e in epochs])

    plot_noise_vs_true_importance(
        epochs,
        importance_matrix,
        feature_names,
        os.path.join(output_dir, "noise_vs_true_importance.png")
    )

    # Select early / mid / late epochs (Could change values later)**********************************************
    selected_epochs = [
        epochs[0],  # early training
        epochs[len(epochs)//2],  #"optimal" training
        epochs[-1]  # logical decay
    ]

    for ep in selected_epochs:
        shap_vals = np.load(os.path.join(output_dir, f"shap_epoch_{ep}.npy"))
        imp = epoch_importance[ep]

        plot_shap_summary(
            shap_vals,
            X_eval.numpy(),
            feature_names,
            os.path.join(output_dir, f"summary_epoch_{ep}.png"),
            f"SHAP Summary — Epoch {ep}"
        )

        plot_shap_bar(
            imp,
            feature_names,
            os.path.join(output_dir, f"bar_epoch_{ep}.png"),
            f"Feature Importance — Epoch {ep}"
        )

        plot_shap_importance_vs_epoch(
            epochs,
            importance_matrix,
            feature_names,
            os.path.join(output_dir, "importance_vs_epoch.png")
        )

    print(f"Results saved to {output_dir}")


def main():
    analyze_run("unregularized")
    analyze_run("l2")
    print("\nSHAP analysis complete.")

if __name__ == "__main__":
    main()