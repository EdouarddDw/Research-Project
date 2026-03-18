"""
Interaction Tracking and Classification via Median-Split PDPs.

Methodology
-----------
For each predetermined interacting feature pair (primary, secondary):
1. Split the validation set into "low" and "high" cohorts by the median of the secondary feature.
2. Compute Partial Dependence of the primary feature over a shared grid for both cohorts.
3. Classify the interaction as:
   - Modifier   : curves differ in steepness but do NOT cross  (fan shape)
   - Crossover  : curves explicitly intersect                  (sign flip)
   - None       : curves are not meaningfully different
4. Apply this classification across saved training snapshots to track when each
   ground-truth interaction is correctly learned and when it degrades under overfitting.
"""

import os
import copy
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from synth import functions as synth_functions
from utils import preprocess_data, set_seed, load_snapshots
from multilayer_perceptron import MLP, train


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Data
    function_index: int = 0          # index into synth.functions (0-based)
    num_samples: int = 3000
    seed: int = 42
    noise_std: float = 0.5

    # Model
    hidden_units: List[int] = field(default_factory=lambda: [256,256])
    use_main_effect_nets: bool = False

    # Training
    nepochs: int = 250
    batch_size: int = 256
    learning_rate: float = 1e-2
    l1_const: float = 0.0
    l2_const: float = 0.0
    early_stopping: bool = False
    patience: int = 10

    # Snapshots
    save_snapshots: bool = True
    snapshot_epochs: Optional[List[int]] = None   # None → default schedule in train()
    snapshot_dir: str = "./snapshots"

    # PDP
    grid_size: int = 50              # number of points along primary-feature grid
    n_background: int = 500          # max background samples for PDP averaging

    # Classification thresholds
    crossover_threshold: float = 0.05   # min fractional overlap area to call Crossover
    modifier_gap_threshold: float = 0.1  # min mean |high - low| to call Modifier

    # Output
    output_dir: str = "./pdp_results1"
    verbose: bool = True


# ---------------------------------------------------------------------------
# PDP utilities
# ---------------------------------------------------------------------------

def compute_split_pdp(
    model: nn.Module,
    X_val: np.ndarray,
    primary_idx: int,
    secondary_idx: int,
    grid_size: int = 50,
    n_background: int = 500,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median-split partial dependence for a (primary, secondary) pair.

    Parameters
    ----------
    model        : Trained nn.Module (eval mode expected by caller).
    X_val        : Validation features, shape (N, p).
    primary_idx  : 0-based column index of the primary feature.
    secondary_idx: 0-based column index of the secondary (splitting) feature.
    grid_size    : Number of evenly spaced points over primary feature range.
    n_background : Number of background samples to average over (speed/memory cap).
    device       : Torch device.

    Returns
    -------
    grid     : 1-D array of primary-feature values (length grid_size).
    pd_low   : PD values for the "low" secondary cohort  (length grid_size).
    pd_high  : PD values for the "high" secondary cohort (length grid_size).
    """
    model.eval()
    median_val = np.median(X_val[:, secondary_idx])

    low_mask  = X_val[:, secondary_idx] <= median_val
    high_mask = X_val[:, secondary_idx] >  median_val

    X_low  = X_val[low_mask]
    X_high = X_val[high_mask]

    # Sub-sample background for efficiency
    rng = np.random.default_rng(0)
    if len(X_low) > n_background:
        X_low  = X_low[rng.choice(len(X_low),  n_background, replace=False)]
    if len(X_high) > n_background:
        X_high = X_high[rng.choice(len(X_high), n_background, replace=False)]

    x_min = X_val[:, primary_idx].min()
    x_max = X_val[:, primary_idx].max()
    grid  = np.linspace(x_min, x_max, grid_size)

    def _partial_dep(X_bg: np.ndarray, grid_vals: np.ndarray) -> np.ndarray:
        pd_vals = np.empty(len(grid_vals))
        X_rep = X_bg.copy()
        with torch.no_grad():
            for gi, g in enumerate(grid_vals):
                X_rep[:, primary_idx] = g
                t = torch.from_numpy(X_rep.astype(np.float32)).to(device)
                preds = model(t).cpu().numpy().squeeze()
                pd_vals[gi] = preds.mean()
        return pd_vals

    pd_low  = _partial_dep(X_low,  grid)
    pd_high = _partial_dep(X_high, grid)

    return grid, pd_low, pd_high


def classify_interaction(
    pd_low: np.ndarray,
    pd_high: np.ndarray,
    crossover_threshold: float = 0.05,
    modifier_gap_threshold: float = 0.1,
) -> str:
    """Classify a split-PDP into Crossover / Modifier / None.

    Parameters
    ----------
    pd_low / pd_high         : PD curves for the two cohorts.
    crossover_threshold      : Fraction of sign-change points above which we
                               declare a Crossover interaction.
    modifier_gap_threshold   : Mean absolute gap required to declare Modifier.

    Returns
    -------
    'Crossover' | 'Modifier' | 'None'
    """
    diff = pd_high - pd_low                         # + means high > low
    sign_changes = np.diff(np.sign(diff))           # non-zero where crossing occurs
    n_crossings = np.count_nonzero(sign_changes)
    crossing_fraction = n_crossings / max(len(diff) - 1, 1)

    mean_gap = np.abs(diff).mean()
    value_range = max(
        np.ptp(pd_low), np.ptp(pd_high), 1e-8
    )
    relative_gap = mean_gap / value_range

    if crossing_fraction >= crossover_threshold:
        return "Crossover"
    elif relative_gap >= modifier_gap_threshold:
        return "Modifier"
    else:
        return "None"


# ---------------------------------------------------------------------------
# Snapshot-level analysis
# ---------------------------------------------------------------------------

def analyse_snapshot(
    state_dict: dict,
    model_template: MLP,
    X_val: np.ndarray,
    pairs_to_check: List[Tuple[int, int]],   # 0-based (primary_idx, secondary_idx)
    cfg: Config,
    device: torch.device,
) -> Dict[Tuple[int, int], dict]:
    """Run split-PDP analysis for all pairs on a single model snapshot.

    Returns
    -------
    results : dict mapping (primary_idx, secondary_idx) ->
              {'grid', 'pd_low', 'pd_high', 'classification'}
    """
    model = copy.deepcopy(model_template)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = {}
    for (pi, si) in pairs_to_check:
        grid, pd_low, pd_high = compute_split_pdp(
            model, X_val, pi, si,
            grid_size=cfg.grid_size,
            n_background=cfg.n_background,
            device=device,
        )
        cls = classify_interaction(
            pd_low, pd_high,
            crossover_threshold=cfg.crossover_threshold,
            modifier_gap_threshold=cfg.modifier_gap_threshold,
        )
        results[(pi, si)] = {
            "grid": grid,
            "pd_low": pd_low,
            "pd_high": pd_high,
            "classification": cls,
        }
    return results


# ---------------------------------------------------------------------------
# Pair generation from ground-truth interactions
# ---------------------------------------------------------------------------

def pairs_from_ground_truth(
    interactions: List[set],
) -> List[Tuple[int, int]]:
    """Generate all ordered (primary, secondary) pairs within each ground-truth
    interaction group. Feature indices are 1-based in synth.py; we convert to 0-based.

    Returns a deduplicated list of (primary_0based, secondary_0based) tuples.
    """
    pairs = set()
    for group in interactions:
        group_0 = [f - 1 for f in group]   # convert to 0-based
        for a, b in itertools.permutations(group_0, 2):
            pairs.add((a, b))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_epoch_pdps(
    epoch: int,
    pair_results: Dict[Tuple[int, int], dict],
    num_features: int,
    ground_truth: List[set],
    save_path: Optional[str] = None,
):
    """Plot all split-PDPs for a single epoch in a grid layout."""
    pairs = sorted(pair_results.keys())
    n = len(pairs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    gt_sets = ground_truth

    for ax_i, (pi, si) in enumerate(pairs):
        ax = axes[ax_i]
        res = pair_results[(pi, si)]

        ax.plot(res["grid"], res["pd_low"],  label=f"x{si+1} low",  color="steelblue", lw=2)
        ax.plot(res["grid"], res["pd_high"], label=f"x{si+1} high", color="tomato",    lw=2)

        # Ground-truth annotation
        pair_set = {pi + 1, si + 1}   # back to 1-based for GT comparison
        in_gt = any(pair_set <= gt for gt in gt_sets)
        gt_label = "✓ GT" if in_gt else ""

        cls = res["classification"]
        color_map = {"Crossover": "crimson", "Modifier": "darkorange", "None": "grey"}
        ax.set_title(
            f"PDP: x{pi+1} | x{si+1}  [{cls}] {gt_label}",
            fontsize=9,
            color=color_map.get(cls, "black"),
        )
        ax.set_xlabel(f"x{pi+1}", fontsize=8)
        ax.set_ylabel("PD", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for ax_i in range(len(pairs), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(f"Split-PDP Analysis — Epoch {epoch}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_classification_timeline(
    timeline: Dict[int, Dict[Tuple[int, int], str]],
    pairs: List[Tuple[int, int]],
    ground_truth: List[set],
    save_path: Optional[str] = None,
):
    """Plot how the classification of each pair evolves across epochs."""
    int_epochs = sorted(k for k in timeline.keys() if isinstance(k, int))
    str_epochs = sorted(k for k in timeline.keys() if isinstance(k, str))
    epochs = int_epochs + str_epochs  # ints first, then "final" etc.
    cls_map = {"Crossover": 2, "Modifier": 1, "None": 0}
    cmap   = plt.cm.get_cmap("RdYlGn", 3)

    fig, ax = plt.subplots(figsize=(max(10, len(epochs) * 0.5), max(4, len(pairs) * 0.6)))

    data = np.zeros((len(pairs), len(epochs)), dtype=int)
    for ei, ep in enumerate(epochs):
        for pi_i, pair in enumerate(pairs):
            cls = timeline[ep].get(pair, "None")
            data[pi_i, ei] = cls_map[cls]

    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pairs)))

    gt_sets = ground_truth
    pair_labels = []
    for (pi, si) in pairs:
        pair_set = {pi + 1, si + 1}
        in_gt = any(pair_set <= gt for gt in gt_sets)
        pair_labels.append(f"x{pi+1}|x{si+1}" + (" ✓" if in_gt else ""))
    ax.set_yticklabels(pair_labels, fontsize=8)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=cmap(0), label="None"),
        Patch(facecolor=cmap(1), label="Modifier"),
        Patch(facecolor=cmap(2), label="Crossover"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_title("Interaction Classification Timeline across Snapshots", fontweight="bold")
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2], label="Classification")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    fn = synth_functions[cfg.function_index]
    X, Y, ground_truth = fn(
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        noise_std=cfg.noise_std,
    )
    if cfg.verbose:
        print(f"[Data] f{cfg.function_index + 1}  X:{X.shape}  GT interactions: {ground_truth}")

    data_loaders = preprocess_data(
        X, Y,
        valid_size=500,
        test_size=500,
        std_scale=True,
        get_torch_loaders=True,
        batch_size=cfg.batch_size,
    )

    # Validation set as numpy for PDP (un-scaled primary feature; we use the
    # standardised values that the trained model sees).
    Xd, _ = preprocess_data(X, Y, valid_size=500, test_size=500, std_scale=True)
    X_val = Xd["val"]

    num_features = X.shape[1]

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model_template = MLP(
        num_features=num_features,
        hidden_units=cfg.hidden_units,
        use_main_effect_nets=cfg.use_main_effect_nets,
    ).to(device)

    # ------------------------------------------------------------------
    # 3. Train with snapshots
    # ------------------------------------------------------------------
    if cfg.verbose:
        print(f"[Train] epochs={cfg.nepochs}  device={device}")

    train_result = train(
        model_template,
        data_loaders,
        nepochs=cfg.nepochs,
        verbose=cfg.verbose,
        early_stopping=cfg.early_stopping,
        patience=cfg.patience,
        l1_const=cfg.l1_const,
        l2_const=cfg.l2_const,
        learning_rate=cfg.learning_rate,
        device=device,
        save_snapshots=cfg.save_snapshots,
        snapshot_epochs=cfg.snapshot_epochs,
        snapshot_dir=cfg.snapshot_dir,
    )

    if cfg.save_snapshots:
        trained_model, test_loss, snapshots = train_result
    else:
        trained_model, test_loss = train_result
        snapshots = {}

    if cfg.verbose:
        print(f"[Train] Final test loss: {test_loss:.4f}")

    # Also analyse the final model as an additional "snapshot"
    snapshots["final"] = copy.deepcopy(trained_model.state_dict())

    # ------------------------------------------------------------------
    # 4. Generate pairs from ground truth
    # ------------------------------------------------------------------
    pairs = pairs_from_ground_truth(ground_truth)
    if cfg.verbose:
        print(f"[PDP] Analysing {len(pairs)} directed pairs across "
              f"{len(snapshots)} snapshots …")

    # ------------------------------------------------------------------
    # 5. Analyse each snapshot
    # ------------------------------------------------------------------
    timeline: Dict = {}      # epoch -> {pair -> classification}
    all_results: Dict = {}   # epoch -> {pair -> full PDP dict}

    sorted_epochs = sorted(
        k for k in snapshots.keys() if isinstance(k, int)
    ) + (["final"] if "final" in snapshots else [])

    for epoch in sorted_epochs:
        if cfg.verbose:
            print(f"  Snapshot epoch={epoch} …", end=" ", flush=True)

        results = analyse_snapshot(
            snapshots[epoch],
            model_template,
            X_val,
            pairs,
            cfg,
            device,
        )
        all_results[epoch] = results
        timeline[epoch] = {pair: results[pair]["classification"] for pair in pairs}

        if cfg.verbose:
            summary = {c: sum(1 for v in timeline[epoch].values() if v == c)
                       for c in ("Crossover", "Modifier", "None")}
            print(summary)

        # Per-epoch PDP plot
        plot_epoch_pdps(
            epoch=epoch,
            pair_results=results,
            num_features=num_features,
            ground_truth=ground_truth,
            save_path=os.path.join(cfg.output_dir, f"pdp_epoch_{epoch}.png"),
        )

    # ------------------------------------------------------------------
    # 6. Timeline visualisation
    # ------------------------------------------------------------------
    plot_classification_timeline(
        timeline=timeline,
        pairs=pairs,
        ground_truth=ground_truth,
        save_path=os.path.join(cfg.output_dir, "classification_timeline.png"),
    )

    # ------------------------------------------------------------------
    # 7. Textual summary: first epoch each GT pair was correctly detected
    # ------------------------------------------------------------------
    print("\n=== Ground-truth Interaction Detection Summary ===")
    for gt_group in ground_truth:
        gt_0 = [f - 1 for f in gt_group]
        relevant_pairs = [
            (pi, si) for (pi, si) in pairs
            if pi in gt_0 and si in gt_0
        ]
        print(f"\nGT group: {{{', '.join(f'x{f}' for f in sorted(gt_group))}}}")
        for pair in relevant_pairs:
            first_detected = None
            first_degraded = None
            prev_cls = None
            for ep in sorted_epochs:
                cls = timeline[ep].get(pair, "None")
                if first_detected is None and cls in ("Modifier", "Crossover"):
                    first_detected = ep
                if (first_detected is not None
                        and first_degraded is None
                        and cls == "None"
                        and prev_cls in ("Modifier", "Crossover")):
                    first_degraded = ep
                prev_cls = cls
            print(
                f"  x{pair[0]+1}|x{pair[1]+1}: "
                f"first detected @ epoch {first_detected}, "
                f"degraded @ epoch {first_degraded}"
            )

    if cfg.verbose:
        print(f"\nResults saved to: {cfg.output_dir}")

    return all_results, timeline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config(
        function_index=0,          # f1 from synth.py
        num_samples=3000,
        nepochs=250,
        hidden_units=[256, 256],
        snapshot_epochs=[1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250],
        save_snapshots=True,
        snapshot_dir="./snapshots",
        output_dir="./pdp_results1",
        verbose=True,
    )
    all_results, timeline = run_pipeline(cfg)
