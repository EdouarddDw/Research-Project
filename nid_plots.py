import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synth
from multilayer_perceptron import MLP, get_weights
from neural_interaction_detection import get_interactions


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_OUT = os.path.join(PROJECT_ROOT, "outputs", "all_functions")
NOISE_LEVELS = [0.1, 0.5]
NUM_FEATURES = 10
HIDDEN_UNITS = [140, 100, 60, 20]
USE_MAIN_EFFECT_NETS = True


def _snapshot_files(search_root: str) -> dict[int, str]:
    """Find model_epoch_*.pt recursively under search_root."""
    if not os.path.isdir(search_root):
        return {}
    pat = re.compile(r"model_epoch_(\d+)\.pt$")
    found = {}
    for root, _, files in os.walk(search_root):
        for fn in files:
            m = pat.match(fn)
            if m:
                ep = int(m.group(1))
                p = os.path.join(root, fn)
                # keep first if duplicate epoch
                found.setdefault(ep, p)
    return dict(sorted(found.items(), key=lambda x: x[0]))


def _gt_pair_set(fn_idx: int) -> set[tuple[int, int]]:
    # synth returns 1-indexed variable ids in interaction sets
    _, _, interactions = synth.functions[fn_idx](num_samples=16, seed=42, noise_std=0.0)
    gt_pairs = set()
    for inter in interactions:
        inter = sorted(int(i) for i in inter)
        if len(inter) < 2:
            continue
        for a, b in combinations(inter, 2):
            gt_pairs.add((a, b))
    return gt_pairs


def build_non_gt_dataframe() -> pd.DataFrame:
    rows = []
    device = torch.device("cpu")

    for noise in NOISE_LEVELS:
        for fn_idx in range(len(synth.functions)):
            noise_root = os.path.join(BASE_OUT, f"F{fn_idx}", f"noise_{noise}")
            snapshots = _snapshot_files(noise_root)
            if not snapshots:
                continue

            gt_pairs = _gt_pair_set(fn_idx)

            model = MLP(
                NUM_FEATURES,
                HIDDEN_UNITS,
                use_main_effect_nets=USE_MAIN_EFFECT_NETS
            ).to(device)
            model.eval()

            for ep, pth in snapshots.items():
                state = torch.load(pth, map_location=device)
                model.load_state_dict(state)
                model.eval()

                pairwise = get_interactions(get_weights(model), pairwise=True, one_indexed=True)

                total_strength = 0.0
                non_gt_strength = 0.0
                for (i, j), s in pairwise:
                    s = float(max(0.0, s))
                    total_strength += s
                    pair = tuple(sorted((int(i), int(j))))
                    if pair not in gt_pairs:
                        non_gt_strength += s

                frac_non_gt = non_gt_strength / (total_strength + 1e-12)
                rows.append({
                    "noise": noise,
                    "function": fn_idx,
                    "epoch": ep,
                    "frac_non_gt_strength": frac_non_gt,
                })

    return pd.DataFrame(rows)


def plot_non_gt_emergence(df: pd.DataFrame, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(BASE_OUT, "non_gt_emergence_noise_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 5))

    for noise, color in [(0.1, "steelblue"), (0.5, "darkorange")]:
        d = df[df["noise"] == noise]
        if d.empty:
            continue
        g = d.groupby("epoch")["frac_non_gt_strength"]
        med = g.median()
        q25 = g.quantile(0.25)
        q75 = g.quantile(0.75)

        ax.plot(med.index, med.values, color=color, linewidth=2,
                label=f"noise={noise} (median)")
        ax.fill_between(med.index, q25.values, q75.values, color=color, alpha=0.20)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction of pairwise interaction strength on non-GT pairs")
    ax.set_title("Emergence of non-ground-truth interactions during training")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    df = build_non_gt_dataframe()
    if df.empty:
        print("No snapshot data found under outputs/all_functions/*/noise_*/snapshots")
        return
    csv_path = os.path.join(BASE_OUT, "non_gt_emergence_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    plot_non_gt_emergence(df)


if __name__ == "__main__":
    main()