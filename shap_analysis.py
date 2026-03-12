"""
analyze_shap_snapshots.py
=========================

Compute SHAP explanations for MLP snapshots trained on synth.functions[3].
"""

import os
import re
import numpy as np
import torch
import shap
from pathlib import Path

import synth
from train_mlp import MLP, SEED, N_SAMPLES, NOISE_STD

from plots import plot_shap_summary, plot_shap_bar, plot_shap_importance_vs_epoch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SNAPSHOT_ROOT = "./outputs/snapshots/f3"
OUTPUT_ROOT   = "./outputs/shap_analysis/f3"

RUN_NAME = "l1"

BACKGROUND_SIZE = 500
EVAL_SIZE       = 1000

HIDDEN = [140, 100, 60, 20]

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ─────────────────────────────────────────────
# MODEL WRAPPER
# ─────────────────────────────────────────────

class ModelWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out


# ─────────────────────────────────────────────
# SNAPSHOT DISCOVERY
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
# SHAP ANALYSIS
# ─────────────────────────────────────────────

def analyze_run(run_name):
    print(f"\nAnalyzing run: {run_name}")
    snapshot_dir = os.path.join(SNAPSHOT_ROOT, run_name)
    output_dir   = os.path.join(OUTPUT_ROOT, run_name)
    os.makedirs(output_dir, exist_ok=True)
    snapshots = discover_snapshots(snapshot_dir)

    if len(snapshots) == 0:
        print(f"No snapshots found in {snapshot_dir}")
        return

    print(f"Found {len(snapshots)} snapshots")

    # SAME DATA GENERATION AS TRAINING
    X, Y, ground_truth = synth.functions[3](num_samples=N_SAMPLES, seed=SEED, noise_std=NOISE_STD)

    feature_names = [f"x{i+1}" for i in range(X.shape[1])]

    # SHAP subsets
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    bg_idx   = idx[:BACKGROUND_SIZE]
    eval_idx = idx[BACKGROUND_SIZE:BACKGROUND_SIZE + EVAL_SIZE]
    X_bg   = torch.tensor(X[bg_idx], dtype=torch.float32)
    X_eval = torch.tensor(X[eval_idx], dtype=torch.float32)

    # Save evaluation data so animation can reuse it for coloring
    np.save(os.path.join(output_dir, "X_eval.npy"), X_eval.numpy())

    epoch_importance = {}
    # ─────────────────────────────────────────
    # LOOP OVER SNAPSHOTS
    # ─────────────────────────────────────────
    for epoch, path in snapshots:
        print(f"  Epoch {epoch}")
        model = MLP(
            num_features=X.shape[1],
            hidden_units=HIDDEN,
            use_main_effect_nets=False
        )
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()

        wrapped_model = ModelWrapper(model)

        explainer = shap.GradientExplainer(wrapped_model, X_bg)

        shap_vals = explainer.shap_values(X_eval)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.array(shap_vals).squeeze()

        importance = np.mean(np.abs(shap_vals), axis=0).flatten()
        importance = importance / np.sum(importance) #normalising values

        epoch_importance[epoch] = importance

        np.save(
            os.path.join(output_dir, f"shap_epoch_{epoch}.npy"),
            shap_vals
        )

    epochs = sorted(epoch_importance.keys())
    importance_matrix = np.vstack(
        [epoch_importance[e] for e in epochs]
    )

    selected_epochs = [
        epochs[0],
        epochs[len(epochs)//2],
        epochs[-1]
    ]

    for ep in selected_epochs:
        shap_vals = np.load(
            os.path.join(output_dir, f"shap_epoch_{ep}.npy")
        )
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
    analyze_run(RUN_NAME)
    print("\nSHAP analysis complete.")


if __name__ == "__main__":
    main()