import os
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Sequence

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
SNAPSHOT_FALLBACK_DIRS = [
    os.path.join(PROJECT_ROOT, "outputs", "snapshots", "animation"),
    os.path.join(PROJECT_ROOT, "outputs", "snapshots", "unregularized"),
    os.path.join(PROJECT_ROOT, "outputs", "snapshots"),
]
NOISE_LEVELS = [0.1, 0.5]
NUM_FEATURES = 10
HIDDEN_UNITS = [140, 100, 60, 20]
USE_MAIN_EFFECT_NETS = True


@dataclass
class NIDPlotConfig:
    project_root: str = PROJECT_ROOT
    base_out: str = BASE_OUT
    noise_levels: tuple[float, ...] = tuple(NOISE_LEVELS)
    num_features: int = NUM_FEATURES
    hidden_units: tuple[int, ...] = tuple(HIDDEN_UNITS)
    use_main_effect_nets: bool = USE_MAIN_EFFECT_NETS
    fallback_snapshot_dirs: tuple[str, ...] = tuple(SNAPSHOT_FALLBACK_DIRS)
    fallback_function: int = 0
    fallback_noise: float = 0.1
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


@dataclass
class SnapshotInteractionMetrics:
    noise: float
    function: int
    epoch: int
    total_strength: float
    non_gt_strength: float
    delta_target_placeholder: float = 0.0
    total_detected_pairs: int = 0
    non_gt_detected_pairs: int = 0
    pct_non_gt_detected_pairs: float = 0.0


class NIDInteractionAnalyzer:
    def __init__(self, config: NIDPlotConfig | None = None):
        self.config = config or NIDPlotConfig()
        self.device = torch.device(self.config.device)

    def snapshot_files(self, search_root: str) -> dict[int, str]:
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
                    found.setdefault(ep, p)
        return dict(sorted(found.items(), key=lambda x: x[0]))

    def gt_pair_set(self, fn_idx: int) -> set[tuple[int, int]]:
        """Return 1 indexed pairwise ground truth interactions for one synthetic function."""
        _, _, interactions = synth.functions[fn_idx](num_samples=16, seed=42, noise_std=0.0)
        gt_pairs = set()
        for inter in interactions:
            inter = sorted(int(i) for i in inter)
            if len(inter) < 2:
                continue
            for a, b in combinations(inter, 2):
                gt_pairs.add((a, b))
        return gt_pairs

    def make_model(self) -> MLP:
        model = MLP(
            self.config.num_features,
            list(self.config.hidden_units),
            use_main_effect_nets=self.config.use_main_effect_nets,
        ).to(self.device)
        model.eval()
        return model

    def load_state_dict_compat(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            return state["model_state_dict"]
        return state

    def compute_pairwise_interactions(self, model: MLP):
        return get_interactions(get_weights(model), pairwise=True, one_indexed=True)

    def compute_snapshot_metric(self, model: MLP, fn_idx: int, noise: float, epoch: int) -> SnapshotInteractionMetrics:
        gt_pairs = self.gt_pair_set(fn_idx)
        pairwise = self.compute_pairwise_interactions(model)

        total_strength = 0.0
        non_gt_strength = 0.0
        total_detected_pairs = 0
        non_gt_detected_pairs = 0

        for (i, j), s in pairwise:
            s = float(max(0.0, s))
            pair = tuple(sorted((int(i), int(j))))

            if s > 0.0:
                total_detected_pairs += 1
                if pair not in gt_pairs:
                    non_gt_detected_pairs += 1

            total_strength += s
            if pair not in gt_pairs:
                non_gt_strength += s

        pct_non_gt_detected_pairs = non_gt_detected_pairs / (total_detected_pairs + 1e-12)
        return SnapshotInteractionMetrics(
            noise=noise,
            function=fn_idx,
            epoch=epoch,
            total_strength=total_strength,
            non_gt_strength=non_gt_strength,
            total_detected_pairs=total_detected_pairs,
            non_gt_detected_pairs=non_gt_detected_pairs,
            pct_non_gt_detected_pairs=pct_non_gt_detected_pairs,
        )

    def build_non_gt_dataframe(self) -> pd.DataFrame:
        rows = []
        found_standard_snapshots = False

        for noise in self.config.noise_levels:
            for fn_idx in range(len(synth.functions)):
                noise_root = os.path.join(self.config.base_out, f"F{fn_idx}", f"noise_{noise}")
                snapshots = self.snapshot_files(noise_root)
                if not snapshots:
                    continue
                found_standard_snapshots = True

                model = self.make_model()

                for ep, pth in snapshots.items():
                    state_dict = self.load_state_dict_compat(pth)
                    model.load_state_dict(state_dict)
                    model.eval()
                    metric = self.compute_snapshot_metric(model, fn_idx, noise, ep)
                    rows.append({
                        "noise": metric.noise,
                        "function": metric.function,
                        "epoch": metric.epoch,
                        "total_strength": metric.total_strength,
                        "non_gt_strength": metric.non_gt_strength,
                        "total_detected_pairs": metric.total_detected_pairs,
                        "non_gt_detected_pairs": metric.non_gt_detected_pairs,
                        "pct_non_gt_detected_pairs": metric.pct_non_gt_detected_pairs,
                    })

        if rows or found_standard_snapshots:
            return pd.DataFrame(rows)

        # Fallback: if per-function/per-noise folders have no snapshots,
        # try generic snapshot directories and compute a single curve using
        # configured defaults for (function, noise).
        for fallback_dir in self.config.fallback_snapshot_dirs:
            snapshots = self.snapshot_files(fallback_dir)
            if not snapshots:
                continue

            print(
                "No snapshots found in outputs/all_functions/*/noise_*. "
                f"Using fallback snapshots from: {fallback_dir} "
                f"with function=F{self.config.fallback_function}, "
                f"noise={self.config.fallback_noise}."
            )

            model = self.make_model()
            for ep, pth in snapshots.items():
                state_dict = self.load_state_dict_compat(pth)
                model.load_state_dict(state_dict)
                model.eval()
                metric = self.compute_snapshot_metric(
                    model,
                    self.config.fallback_function,
                    self.config.fallback_noise,
                    ep,
                )
                rows.append({
                    "noise": metric.noise,
                    "function": metric.function,
                    "epoch": metric.epoch,
                    "total_strength": metric.total_strength,
                    "non_gt_strength": metric.non_gt_strength,
                    "total_detected_pairs": metric.total_detected_pairs,
                    "non_gt_detected_pairs": metric.non_gt_detected_pairs,
                    "pct_non_gt_detected_pairs": metric.pct_non_gt_detected_pairs,
                })
            break

        return pd.DataFrame(rows)


def plot_non_gt_emergence(
    df: pd.DataFrame,
    save_path: str | None = None,
    title: str = "Percentage of detected pairwise interactions not in ground truth during training",
    ax: plt.Axes | None = None,
):
    if save_path is None:
        save_path = os.path.join(BASE_OUT, "non_gt_emergence_noise_comparison.png")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.figure

    for noise, color in [(0.1, "steelblue"), (0.5, "darkorange")]:
        d = df[df["noise"] == noise]
        if d.empty:
            continue
        g = d.groupby("epoch")["pct_non_gt_detected_pairs"]
        med = g.median()
        q25 = g.quantile(0.25)
        q75 = g.quantile(0.75)

        ax.plot(med.index, med.values, color=color, linewidth=2,
                label=f"noise={noise} (median)")
        ax.fill_between(med.index, q25.values, q75.values, color=color, alpha=0.20)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percentage of detected pairwise interactions not in ground truth")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")

    return fig, ax


# Plotting delta of non-GT emergence
def plot_non_gt_emergence_delta(
    df: pd.DataFrame,
    save_path: str | None = None,
    title: str = "Epoch to epoch increase in non-ground-truth interaction strength",
    ax: plt.Axes | None = None,
):
    if save_path is None:
        save_path = os.path.join(BASE_OUT, "non_gt_emergence_delta_noise_comparison.png")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        created_fig = True
    else:
        fig = ax.figure

    for noise, color in [(0.1, "steelblue"), (0.5, "darkorange")]:
        d = df[df["noise"] == noise].copy()
        if d.empty:
            continue

        delta_rows = []

        for fn_idx, fn_df in d.groupby("function"):
            fn_series = fn_df.groupby("epoch")["non_gt_strength"].median().sort_index()
            deltas = fn_series.diff()
            for epoch, delta in deltas.items():
                if pd.isna(delta):
                    continue
                delta_rows.append({
                    "function": fn_idx,
                    "epoch": epoch,
                    "delta_non_gt_strength": float(delta),
                })

        if not delta_rows:
            continue

        delta_df = pd.DataFrame(delta_rows)
        g = delta_df.groupby("epoch")["delta_non_gt_strength"]
        med = g.median()
        q25 = g.quantile(0.25)
        q75 = g.quantile(0.75)

        ax.plot(
            med.index,
            med.values,
            color=color,
            linewidth=2,
            label=f"noise={noise} (median)",
        )
        ax.fill_between(med.index, q25.values, q75.values, color=color, alpha=0.20)

    ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ non-GT pairwise interaction strength")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")

    return fig, ax


def build_non_gt_dataframe(config: NIDPlotConfig | None = None) -> pd.DataFrame:
    analyzer = NIDInteractionAnalyzer(config=config)
    return analyzer.build_non_gt_dataframe()



def compute_non_gt_fraction_for_model(
    model: MLP,
    fn_idx: int,
    noise: float,
    analyzer: NIDInteractionAnalyzer | None = None,
) -> dict:
    analyzer = analyzer or NIDInteractionAnalyzer()
    metric = analyzer.compute_snapshot_metric(model, fn_idx, noise, epoch=-1)
    return {
        "noise": metric.noise,
        "function": metric.function,
        "epoch": metric.epoch,
        "total_strength": metric.total_strength,
        "non_gt_strength": metric.non_gt_strength,
        "total_detected_pairs": metric.total_detected_pairs,
        "non_gt_detected_pairs": metric.non_gt_detected_pairs,
        "pct_non_gt_detected_pairs": metric.pct_non_gt_detected_pairs,
    }



def add_non_gt_fraction_annotation(
    ax: plt.Axes,
    model: MLP,
    fn_idx: int,
    noise: float,
    analyzer: NIDInteractionAnalyzer | None = None,
    x: float = 0.02,
    y: float = 0.98,
):
    metrics = compute_non_gt_fraction_for_model(model, fn_idx, noise, analyzer=analyzer)
    ax.text(
        x,
        y,
        f"non-GT detected pairs: {metrics['pct_no_gt_detected_pairs']:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    return metrics


def main():
    config = NIDPlotConfig()
    df = build_non_gt_dataframe(config=config)
    if df.empty:
        print("No snapshot data found under outputs/all_functions/*/noise_*")
        return
    csv_path = os.path.join(config.base_out, "non_gt_emergence_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    plot_non_gt_emergence(
        df,
        save_path=os.path.join(config.base_out, "non_gt_emergence_noise_comparison.png"),
    )

    plot_non_gt_emergence_delta(
        df,
        save_path=os.path.join(config.base_out, "non_gt_emergence_delta_noise_comparison.png"),
    )


if __name__ == "__main__":
    main()
