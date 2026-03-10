from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

# ── Colab detection & backend ─────────────────────────────────────────────────
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ
if not IN_COLAB:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence

# ── Project imports ───────────────────────────────────────────────────────────
import networkx as nx

import synth
from utils import set_seed

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ── Paths ─────────────────────────────────────────────────────────────────────
if IN_COLAB:
    BASE_DIR = Path("/content")
else:
    BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "shap_outputs"

# ── Config ────────────────────────────────────────────────────────────────────
# Synthetic benchmark function index (0=F1…6=F6).  Default 3 = F3 (★).
SYNTH_FUNC_IDX    = 3
NUM_SAMPLES       = 1_500
SEED              = 42

SNAPSHOT_EPOCHS   = [1, 3, 5, 10, 20, 30, 40, 50, 100, 150, 250]
INTERACTION_EPOCHS = [20, 50, 150]
TOTAL_EPOCHS      = 250
BATCH_SIZE        = 128
LR                = 1e-4
N_SHAP_BG         = 80    # KernelExplainer background rows
N_SHAP_EXPLAIN    = 60    # rows explained per snapshot
PDP_SUBSAMPLE     = 300
DPI               = 130

HIDDEN_DIMS       = [128, 64, 32]
DROPOUT           = 0.0
WEIGHT_DECAY      = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Runtime globals (populated by load_synth_data) ────────────────────────────
FEATURE_COLS:    list[str] = []
SIGNAL_FEATURES: list[str] = []
NOISE_FEATURES:  list[str] = []
GT_PAIRS:        list[tuple[str, str]] = []
CONTRAST_PAIRS:  list[tuple[str, str, str]] = []   # (f1, f2, label) where label ∈ {signal, mixed, noise}

_T0 = time.perf_counter()
def _elapsed() -> str:
    s = int(time.perf_counter() - _T0)
    m, s = divmod(s, 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"

def ph(title: str) -> None:
    print(f"\n{'='*80}\n[{_elapsed()}]  {title}\n{'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class PurchaseNN(nn.Module):
    """
    Configurable MLP for binary purchase prediction.

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dims : list of hidden layer widths, e.g. [64, 32]
    dropout     : dropout probability (0.0 = no dropout → overfit)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # logits, shape (N,)


class ModelWrapper(ClassifierMixin, BaseEstimator):
    """
    Wraps a trained PurchaseNN so it exposes a sklearn-style
    ``predict_proba(X_numpy) -> proba_class1`` interface required by
    SHAP KernelExplainer and sklearn PDP.
    Internal StandardScaler is stored so raw (unscaled) input can be passed.
    """

    _estimator_type = "classifier"

    def __init__(self, model: PurchaseNN = None, scaler: StandardScaler = None) -> None:
        self.model  = model
        self.scaler = scaler
        # sklearn duck-typing: attributes ending with '_' signal "fitted"
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True

    # ── sklearn PDP / SHAP predict_proba interface ────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xs = self.scaler.transform(X)
        t  = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(t).cpu().numpy()
        proba1 = 1.0 / (1.0 + np.exp(-logits))           # sigmoid
        return np.column_stack([1 - proba1, proba1])

    # needed by PartialDependenceDisplay
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # sklearn duck-typing helpers
    def fit(self, *_):  return self
    def get_params(self, **_): return {}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SNAPSHOT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Snapshot:
    epoch:        int
    train_loss:   float
    valid_loss:   float
    train_auc:    float
    valid_auc:    float

    # optional — filled only at SNAPSHOT_EPOCHS
    shap_values:  Optional[np.ndarray]  = None   # (N_SHAP_EXPLAIN, n_features)
    # optional — filled only at INTERACTION_EPOCHS
    interaction_proxy: Optional[np.ndarray] = None  # (n_features, n_features)
    model_state:  Optional[dict]        = None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING & PREPROCESSING  (synth.py → binarised classification)
# ══════════════════════════════════════════════════════════════════════════════

def _derive_feature_metadata(ground_truth: dict, n_features: int) -> None:
    """
    Populate the module-level FEATURE_COLS, SIGNAL_FEATURES, NOISE_FEATURES,
    and GT_PAIRS from synth.py's ground_truth dict.
    """
    global FEATURE_COLS, SIGNAL_FEATURES, NOISE_FEATURES, GT_PAIRS, CONTRAST_PAIRS

    FEATURE_COLS = [f"x{i}" for i in range(n_features)]

    # Signal = any feature that appears in at least one ground-truth tuple
    signal_idx = sorted({i for tup in ground_truth["pairwise"] + ground_truth["any_order"]
                         for i in tup})
    noise_idx  = sorted(set(range(n_features)) - set(signal_idx))

    SIGNAL_FEATURES = [FEATURE_COLS[i] for i in signal_idx]
    NOISE_FEATURES  = [FEATURE_COLS[i] for i in noise_idx]

    # GT_PAIRS: take the pairwise ground-truth tuples (0-indexed ints → name strings)
    GT_PAIRS = [(FEATURE_COLS[a], FEATURE_COLS[b])
                for (a, b) in ground_truth["pairwise"]]

    gt_set = {(a, b) for a, b in GT_PAIRS}

    # CONTRAST_PAIRS: ALL signal×signal combos (labelled "signal" if GT, else "signal_non_gt"),
    #                 + signal×noise ("mixed") + noise×noise ("noise")
    CONTRAST_PAIRS = []

    # All signal×signal combinations
    for f1, f2 in combinations(SIGNAL_FEATURES, 2):
        if (f1, f2) in gt_set or (f2, f1) in gt_set:
            CONTRAST_PAIRS.append((f1, f2, "signal"))
        else:
            CONTRAST_PAIRS.append((f1, f2, "signal"))  # still signal features

    # Add up to 4 signal×noise mixed pairs (if noise features exist)
    mixed_count = 0
    for sf in SIGNAL_FEATURES:
        for nf in NOISE_FEATURES:
            CONTRAST_PAIRS.append((sf, nf, "mixed"))
            mixed_count += 1
            if mixed_count >= 4:
                break
        if mixed_count >= 4:
            break

    # Add up to 2 noise×noise pairs
    noise_combos = list(combinations(NOISE_FEATURES, 2))[:2]
    CONTRAST_PAIRS += [(f1, f2, "noise") for f1, f2 in noise_combos]


def load_synth_data(
    func_idx: int   = SYNTH_FUNC_IDX,
    num_samples: int = NUM_SAMPLES,
    seed: int       = SEED,
    valid_size: int = 500,
    test_size: int  = 500,
):
    """
    Generate data from synth.functions[func_idx], binarise Y at the median,
    and split into train / val / test.

    Returns
    -------
    X_tr, y_tr, X_va, y_va, X_te, y_te : scaled arrays  (float32)
    X_va_raw, X_tr_raw                  : raw (unscaled) arrays for SHAP/PDP
    scaler                              : fitted StandardScaler
    X_va_df                             : pd.DataFrame of raw validation data
    ground_truth                        : dict from synth.py
    """
    X, Y_cont, ground_truth = synth.functions[func_idx](
        num_samples=num_samples, seed=seed
    )

    # Binarise regression target at the median
    Y = (Y_cont > np.median(Y_cont)).astype(np.float32)

    n = len(X)
    n_features = X.shape[1]
    _derive_feature_metadata(ground_truth, n_features)

    # Split: train / val / test
    n_train = n - valid_size - test_size
    X_tr_raw = X[:n_train].astype(np.float32)
    X_va_raw = X[n_train:n_train + valid_size].astype(np.float32)
    X_te_raw = X[n_train + valid_size:].astype(np.float32)

    y_tr = Y[:n_train]
    y_va = Y[n_train:n_train + valid_size]
    y_te = Y[n_train + valid_size:]

    # Fit StandardScaler on train (synth data is [0,1] so effect is mild,
    # but keeps the ModelWrapper interface consistent)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_va = scaler.transform(X_va_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)

    # DataFrame for PDP / conditional PDP (uses raw values + feature names)
    X_va_df = pd.DataFrame(X_va_raw, columns=FEATURE_COLS)

    return (X_tr, y_tr, X_va, y_va, X_te, y_te,
            X_va_raw, X_tr_raw, scaler, X_va_df, ground_truth)


def make_loader(X, y, shuffle=True):
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(y)
    )
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP  (with snapshot collection)
# ══════════════════════════════════════════════════════════════════════════════

def compute_interaction_proxy(
    shap_vals: np.ndarray,
    X_raw: np.ndarray,
) -> np.ndarray:
    """
    Proxy for feature interaction: |corr(SHAP_i, X_j)| for every pair (i,j).
    Shape → (n_features, n_features).
    Non-diagonal entries capture how much feature j's raw value is linearly
    associated with feature i's SHAP contribution → asymmetric modifier signal.
    """
    n_feat = shap_vals.shape[1]
    proxy  = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(n_feat):
            if i != j:
                c = np.corrcoef(shap_vals[:, i], X_raw[:, j])[0, 1]
                proxy[i, j] = abs(c) if np.isfinite(c) else 0.0
    return proxy


def compute_h_statistic(
    wrapper: ModelWrapper,
    X_pdp: pd.DataFrame,
    f1_idx: int,
    f2_idx: int,
    grid_res: int = 30,
) -> float:
    """
    Friedman & Popescu (2008) H-statistic for the pair (f1_idx, f2_idx).

    All PDPs must be **centred** (subtract the overall mean prediction f̄)
    before computing the ratio:

        H²_jk = Σ [f̃_jk(x_j, x_k) - f̃_j(x_j) - f̃_k(x_k)]²
                / Σ f̃_jk(x_j, x_k)²

    where f̃ = PDP − f̄.

    Returns H ∈ [0, 1].  Values near 0 → additive, near 1 → strong interaction.
    """
    try:
        pdp_j  = partial_dependence(wrapper, X_pdp, features=[f1_idx],
                                    grid_resolution=grid_res, kind="average")
        pdp_k  = partial_dependence(wrapper, X_pdp, features=[f2_idx],
                                    grid_resolution=grid_res, kind="average")
        pdp_jk = partial_dependence(wrapper, X_pdp, features=[(f1_idx, f2_idx)],
                                    grid_resolution=grid_res, kind="average")

        # Raw PDP arrays (NOT yet centred)
        pdp_j_vals  = pdp_j["average"][0]       # shape (grid_j,)
        pdp_k_vals  = pdp_k["average"][0]       # shape (grid_k,)
        pdp_jk_vals = pdp_jk["average"][0]      # shape (grid_j, grid_k)

        # Overall mean prediction (f̄) — average of the 2-D PDP surface
        f_bar = pdp_jk_vals.mean()

        # Centre everything by subtracting f̄
        pdp_j_c  = pdp_j_vals  - f_bar           # f̃_j
        pdp_k_c  = pdp_k_vals  - f_bar           # f̃_k
        pdp_jk_c = pdp_jk_vals - f_bar           # f̃_jk

        # Broadcast centred marginals onto the 2-D grid
        pdp_j_2d = pdp_j_c[:, None]              # (grid_j, 1)
        pdp_k_2d = pdp_k_c[None, :]              # (1, grid_k)

        # Numerator: interaction residual after removing centred main effects
        residual = pdp_jk_c - pdp_j_2d - pdp_k_2d
        numer    = np.sum(residual ** 2)

        # Denominator: total centred joint variance
        denom = np.sum(pdp_jk_c ** 2)
        if denom < 1e-20:
            return 0.0

        h_sq = numer / denom
        return float(np.sqrt(np.clip(h_sq, 0, 1)))
    except Exception:
        return 0.0


def train_model(
    n_features: int,
    hidden_dims: list[int],
    dropout: float,
    weight_decay: float,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_bg: np.ndarray,
    X_ex: np.ndarray,
    scaler: StandardScaler,
    model_name: str = "model",
) -> tuple[ModelWrapper, list[Snapshot]]:

    ph(f"Training  {model_name}  |  layers={hidden_dims}  dropout={dropout}  wd={weight_decay}")

    model     = PurchaseNN(n_features, hidden_dims, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader    = make_loader(X_tr, y_tr, shuffle=True)
    wrapper   = ModelWrapper(model, scaler)

    snapshots: list[Snapshot] = []

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(X_tr)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            va_logits = model(torch.tensor(X_va).to(DEVICE)).cpu().numpy()
            va_loss   = criterion(
                torch.tensor(va_logits),
                torch.tensor(y_va)
            ).item()
            tr_logits = model(torch.tensor(X_tr).to(DEVICE)).cpu().numpy()

        tr_proba = 1 / (1 + np.exp(-tr_logits))
        va_proba = 1 / (1 + np.exp(-va_logits))
        tr_auc   = roc_auc_score(y_tr, tr_proba)
        va_auc   = roc_auc_score(y_va, va_proba)

        snap = Snapshot(epoch, tr_loss, va_loss, tr_auc, va_auc)

        # ── SHAP snapshot ─────────────────────────────────────────────────────
        if epoch in SNAPSHOT_EPOCHS:
            print(f"  epoch {epoch:3d}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}"
                  f"  va_auc={va_auc:.4f}  → computing SHAP …", end=" ", flush=True)
            explainer   = shap.KernelExplainer(
                lambda x: wrapper.predict_proba(x)[:, 1], X_bg
            )
            shap_vals   = explainer.shap_values(X_ex, nsamples=80, silent=True)
            snap.shap_values = np.asarray(shap_vals)
            print(f"done [{_elapsed()}]")

            # ── Interaction proxy ─────────────────────────────────────────────
            if epoch in INTERACTION_EPOCHS:
                snap.interaction_proxy = compute_interaction_proxy(
                    snap.shap_values, X_ex
                )
                snap.model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  interaction proxy computed  [{_elapsed()}]")
        else:
            if epoch % 25 == 0:
                print(f"  epoch {epoch:3d}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}"
                      f"  va_auc={va_auc:.4f}")

        snapshots.append(snap)

    return wrapper, snapshots


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INTERACTION TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_interaction(
    shap_vals: np.ndarray,
    X_raw: np.ndarray,
    f1: str,
    f2: str,
) -> str:
    """
    Classify the interaction between f1 and f2 using *slope-based* logic:

    1. Split observations at the median of f2 into low / high subgroups.
    2. In each subgroup, fit a linear slope:  SHAP(f1) ~ X(f1).
       The slope captures the *direction* of f1's effect within that group.
    3. Decision rules:
       - **crossover**:  slopes have opposite signs (the effect of f1
         reverses depending on whether f2 is low or high).
       - **modifier**:   slopes share the same sign but ratio > 2.0
         (f2 heavily amplifies / dampens f1's effect).
       - **additive**:   slopes share the same sign and ratio < 1.2
         (nearly parallel lines → purely independent, zero interaction).
       - **symmetric**:  catch-all for ratios between 1.2 and 2.0
         (mutual but moderate interaction).
    """
    i1 = FEATURE_COLS.index(f1)
    i2 = FEATURE_COLS.index(f2)

    shap_f1 = shap_vals[:, i1]
    x_f1    = X_raw[:, i1]
    x_f2    = X_raw[:, i2]

    median_f2 = np.median(x_f2)
    low_mask  = x_f2 <= median_f2
    high_mask = x_f2 > median_f2

    if low_mask.sum() < 5 or high_mask.sum() < 5:
        return "insufficient_data"

    # ── fit linear slope in each subgroup ─────────────────────────────────
    def _slope(x, y):
        """OLS slope (single regressor)."""
        if len(x) < 3 or np.std(x) < 1e-12:
            return 0.0
        return np.polyfit(x, y, 1)[0]          # coefficient of x^1

    slope_low  = _slope(x_f1[low_mask],  shap_f1[low_mask])
    slope_high = _slope(x_f1[high_mask], shap_f1[high_mask])

    # ── decision rules ────────────────────────────────────────────────────
    # Crossover: slopes point in opposite directions
    if slope_low * slope_high < 0:
        return "crossover"

    # Ratio of the larger absolute slope to the smaller absolute slope
    abs_lo = abs(slope_low)
    abs_hi = abs(slope_high)
    ratio  = max(abs_lo, abs_hi) / max(min(abs_lo, abs_hi), 1e-12)

    # Modifier: one feature heavily amplifies/dampens the other
    if ratio > 2.0:
        return "modifier"

    # Additive: nearly parallel lines → purely independent, zero interaction
    if ratio < 1.2:
        return "additive"

    # Symmetric: moderate mutual interaction (ratio between 1.2 and 2.0)
    return "symmetric"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 6a  Training curves ───────────────────────────────────────────────────────
def plot_training_curves(
    snaps: list[Snapshot],
    model_label: str = "model",
) -> None:
    epochs = [s.epoch for s in snaps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, [s.train_loss for s in snaps], "b-",  label="train")
    axes[0].plot(epochs, [s.valid_loss for s in snaps], "b--", label="valid")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title(f"{model_label} – Training vs Validation Loss"); axes[0].legend()

    # AUC
    axes[1].plot(epochs, [s.train_auc for s in snaps], "b-",  label="train")
    axes[1].plot(epochs, [s.valid_auc for s in snaps], "b--", label="valid")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title(f"{model_label} – Training vs Validation AUC"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_training_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved: 01_training_curves.png")


# ── 6b  Feature importance evolution heatmap ─────────────────────────────────
def plot_importance_evolution(
    snaps: list[Snapshot],
    model_label: str,
) -> None:
    shap_snaps = [s for s in snaps if s.shap_values is not None]
    epochs     = [s.epoch for s in shap_snaps]
    # matrix: rows = features, cols = snapshot epochs
    mat = np.array([
        np.abs(s.shap_values).mean(axis=0) for s in shap_snaps
    ]).T  # (n_features, n_epochs)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(
        pd.DataFrame(mat, index=FEATURE_COLS, columns=epochs),
        cmap="YlOrRd", ax=ax, linewidths=0.3,
    )
    ax.set_title(f"{model_label} – Feature importance evolution (mean |SHAP|)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Feature")
    plt.tight_layout()
    fname = f"02_importance_evolution_{model_label}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ── 6c  Noise inflation ──────────────────────────────────────────────────────
def plot_noise_inflation(
    snaps: list[Snapshot],
    model_label: str = "model",
) -> None:
    def _mean_shap_group(snaps, cols):
        result = []
        for s in snaps:
            if s.shap_values is not None:
                idx  = [FEATURE_COLS.index(c) for c in cols]
                result.append((s.epoch, np.abs(s.shap_values[:, idx]).mean()))
        return zip(*result) if result else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))

    sig_e, sig_v = _mean_shap_group(snaps, SIGNAL_FEATURES)
    noi_e, noi_v = _mean_shap_group(snaps, NOISE_FEATURES)
    ax.plot(list(sig_e), list(sig_v), color="steelblue", lw=2, label="Signal features")
    ax.plot(list(noi_e), list(noi_v), color="tomato",    lw=2,
            linestyle="--", label="Noise features")
    ax.set_title(f"{model_label} – Signal vs Noise SHAP")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mean |SHAP|")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_noise_inflation.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved: 03_noise_inflation.png")


# ── 6d  Beeswarm at early / optimal / late ───────────────────────────────────
def plot_beeswarms(
    snaps: list[Snapshot],
    X_ex_raw: np.ndarray,
    model_label: str,
) -> None:
    shap_snaps = [s for s in snaps if s.shap_values is not None]
    # pick early (first), middle (closest to 50), late (last)
    target_epochs = [SNAPSHOT_EPOCHS[0], 50, SNAPSHOT_EPOCHS[-1]]
    chosen = []
    for te in target_epochs:
        best = min(shap_snaps, key=lambda s: abs(s.epoch - te))
        chosen.append(best)

    labels = ["Early", "Optimal", "Late (overfit)"]

    # Render each beeswarm to its own standalone figure, then tile into one
    # image via Pillow — avoids the label/colorbar overlap that occurs when
    # SHAP beeswarm shares a tight subplot grid.
    from PIL import Image
    import io

    cell_images: list[Image.Image] = []
    for snap, lbl in zip(chosen, labels):
        expl = shap.Explanation(
            values=snap.shap_values,
            data=X_ex_raw,
            feature_names=FEATURE_COLS,
        )
        tmp_fig = plt.figure(figsize=(7, 6))
        shap.plots.beeswarm(expl, max_display=12, show=False)
        plt.title(f"{model_label} – {lbl} (epoch {snap.epoch})",
                  fontsize=11, pad=8)
        plt.tight_layout()
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(tmp_fig)
        buf.seek(0)
        cell_images.append(Image.open(buf).copy())
        buf.close()

    # Tile horizontally
    pad = 20
    cw = max(img.width for img in cell_images)
    ch = max(img.height for img in cell_images)
    n_cells = len(cell_images)
    canvas_w = n_cells * cw + (n_cells - 1) * pad
    canvas_h = ch
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    for idx, img in enumerate(cell_images):
        x = idx * (cw + pad) + (cw - img.width) // 2
        y = (ch - img.height) // 2
        canvas.paste(img, (x, y))

    fname = f"04_beeswarms_{model_label}.png"
    canvas.save(str(OUTPUT_DIR / fname))
    for img in cell_images:
        img.close()
    print(f"  saved: {fname}")


# ── 6e  Interaction proxy heatmaps at 3 epochs ───────────────────────────────
def plot_interaction_proxy_heatmaps(
    snaps: list[Snapshot],
    model_label: str,
) -> None:
    proxy_snaps = [s for s in snaps if s.interaction_proxy is not None]
    n = len(proxy_snaps)
    if n == 0:
        print("  [warn] no interaction proxy snapshots found, skipping.")
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    n_show = min(len(FEATURE_COLS), 10)
    for ax, snap in zip(axes, proxy_snaps):
        df = pd.DataFrame(
            snap.interaction_proxy[:n_show, :n_show],
            index=FEATURE_COLS[:n_show],
            columns=FEATURE_COLS[:n_show],
        )
        sns.heatmap(df, cmap="coolwarm", vmin=0, vmax=1,
                    ax=ax, linewidths=0.3, square=True)
        ax.set_title(f"Epoch {snap.epoch}\n|corr(SHAP_i, X_j)|")

    fig.suptitle(f"{model_label} – Interaction proxy evolution", fontsize=13, y=1.02)
    plt.tight_layout()
    fname = f"05_interaction_proxy_{model_label}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ── 6f  Interaction evolution line plot (contrast pairs) ─────────────────────
_PAIR_STYLE = {
    "signal": dict(color="#2ca02c", marker="o", linewidth=2.5, linestyle="-"),
    "mixed":  dict(color="#ff7f0e", marker="s", linewidth=2.0, linestyle="--"),
    "noise":  dict(color="#d62728", marker="^", linewidth=2.0, linestyle=":"),
}

def plot_interaction_evolution(
    snaps: list[Snapshot],
    X_ex_raw: np.ndarray,
    model_label: str = "model",
) -> None:
    """
    Track interaction proxy |corr(SHAP_f1, X_f2)| across epochs for
    three categories of feature pairs:
      • signal (GT pairs)  – should spike early and stay high
      • cross  (signal×noise) – should stay low, may creep up with overfitting
      • noise  (noise×noise)  – should stay near zero, grows late if overfitting
    All curves on one axis for a clear contrast.
    """
    if not CONTRAST_PAIRS:
        print("  [warn] no CONTRAST_PAIRS, skipping interaction evolution.")
        return

    shap_snaps = [s for s in snaps if s.shap_values is not None]
    if not shap_snaps:
        print("  [warn] no SHAP snapshots, skipping interaction evolution.")
        return

    epochs = [s.epoch for s in shap_snaps]

    fig, ax = plt.subplots(figsize=(12, 5))

    for f1, f2, cat in CONTRAST_PAIRS:
        i1 = FEATURE_COLS.index(f1)
        i2 = FEATURE_COLS.index(f2)

        proxies = []
        for s in shap_snaps:
            c = np.corrcoef(s.shap_values[:, i1], X_ex_raw[:, i2])[0, 1]
            proxies.append(abs(c) if np.isfinite(c) else 0.0)

        style = _PAIR_STYLE[cat]
        ax.plot(epochs, proxies, label=f"{f1}×{f2} [{cat}]", **style)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("|corr(SHAP_i, X_j)|  (interaction proxy)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(f"{model_label} – Interaction proxy evolution  "
                 "(signal vs cross vs noise pairs)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_interaction_evolution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  saved: 06_interaction_evolution.png")


# ── 6g  Interaction type classification + conditioned 1-D PDPs ───────────────
def plot_interaction_classification(
    wrapper: ModelWrapper,
    snaps: list[Snapshot],
    X_ex_raw: np.ndarray,
    X_va_df: pd.DataFrame,
    model_label: str,
) -> None:
    """
    For each *contrast* pair (signal, cross, noise): classify interaction type
    (crossover / modifier / symmetric) using the LAST snapshot, then plot
    SHAP scatter + conditional 1-D PDP.
    Signal pairs should show clear interaction; noise pairs should show
    false-positive interactions only when overfitting.
    """
    last_shap = next(
        (s for s in reversed(snaps) if s.shap_values is not None), None
    )
    if last_shap is None:
        print("  [warn] no SHAP snapshots, skipping interaction classification.")
        return

    pairs = CONTRAST_PAIRS if CONTRAST_PAIRS else [(f1, f2, "signal") for f1, f2 in GT_PAIRS]
    n_pairs = len(pairs)
    if n_pairs == 0:
        print("  [warn] no pairs for interaction classification.")
        return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 5 * n_pairs),
                              squeeze=False)

    cat_colours = {"signal": "#2ca02c", "mixed": "#ff7f0e", "noise": "#d62728"}

    for row, (f1, f2, cat) in enumerate(pairs):
        i1 = FEATURE_COLS.index(f1)
        i2 = FEATURE_COLS.index(f2)
        itype = classify_interaction(last_shap.shap_values, X_ex_raw, f1, f2)

        # ── left panel: SHAP scatter coloured by f2 ──────────────────────────
        ax_l = axes[row][0]
        sc   = ax_l.scatter(
            X_ex_raw[:, i1],
            last_shap.shap_values[:, i1],
            c=X_ex_raw[:, i2],
            cmap="coolwarm", alpha=0.5, s=15, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax_l, label=f2)
        ax_l.axhline(0, color="k", lw=0.8, linestyle="--")
        ax_l.set_xlabel(f1); ax_l.set_ylabel(f"SHAP({f1})")
        badge = cat.upper()
        ax_l.set_title(f"[{badge} · {itype.upper()}]  {f1} (colour = {f2})",
                       color=cat_colours.get(cat, "k"))

        # ── right panel: conditioned 1-D PDP ─────────────────────────────────
        ax_r   = axes[row][1]
        median = X_pdp[f2].median()
        X_low  = X_pdp[X_pdp[f2] <= median]
        X_high = X_pdp[X_pdp[f2] > median]

        f1_idx = FEATURE_COLS.index(f1)

        full_col = X_pdp.iloc[:, f1_idx].values
        shared_grid = np.linspace(
            np.percentile(full_col, 2),
            np.percentile(full_col, 98),
            50,
        )

        for subset, colour, lbl in [
            (X_low,  "steelblue", f"{f2} low"),
            (X_high, "tomato",    f"{f2} high"),
        ]:
            if len(subset) > 10:
                tile = np.tile(subset.values, (len(shared_grid), 1, 1))
                tile[:, :, f1_idx] = shared_grid[:, None]
                G, N, F = tile.shape
                flat = tile.reshape(G * N, F)
                proba = wrapper.predict_proba(flat)[:, 1].reshape(G, N)
                pdp_vals = proba.mean(axis=1)
                ax_r.plot(
                    shared_grid, pdp_vals,
                    color=colour, linewidth=2, label=lbl,
                )
        ax_r.set_xlabel(f1)
        ax_r.set_ylabel("Partial dependence")
        ax_r.set_title(f"PDP of {f1} | {f2} split  [{badge} · {itype}]",
                       color=cat_colours.get(cat, "k"))
        ax_r.legend()

    fig.suptitle(f"{model_label} – Interaction classification (signal vs cross vs noise)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f"07_interaction_classification_{model_label}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ── 6h  2-D PDP contour plots at 3 interaction epochs ───────────────────────
def plot_2d_pdp_at_epochs(
    model_name: str,
    snaps: list[Snapshot],
    hidden_dims: list[int],
    dropout: float,
    scaler: StandardScaler,
    X_va_df: pd.DataFrame,
) -> None:
    """
    For each epoch in INTERACTION_EPOCHS, reload the saved model state,
    build a wrapper, and render a 2×2 grid of 2-D PDP contour plots.
    """
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping 2-D PDP at epochs.")
        return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)
    pairs = GT_PAIRS[:4]

    for snap in proxy_snaps:
        # Reconstruct model from saved state
        m = PurchaseNN(len(FEATURE_COLS), hidden_dims, dropout)
        m.load_state_dict(snap.model_state)
        m.eval()
        w = ModelWrapper(m, scaler)

        cols = 2
        rows = int(np.ceil(len(pairs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 8 * rows))
        axes = np.array(axes).flatten()

        for ax, (f1, f2) in zip(axes, pairs):
            i1 = FEATURE_COLS.index(f1)
            i2 = FEATURE_COLS.index(f2)
            pdp_res = partial_dependence(
                w, X_pdp, features=[(i1, i2)],
                grid_resolution=30, kind="average",
            )
            g1, g2 = pdp_res["grid_values"]
            Z = pdp_res["average"][0]
            G1, G2 = np.meshgrid(g1, g2)
            cs = ax.contourf(G1, G2, Z.T, levels=20, cmap="viridis", alpha=0.8)
            fig.colorbar(cs, ax=ax)
            ax.set_xlabel(f1); ax.set_ylabel(f2)
            ax.set_title(f"{f1} × {f2}")

        for ax in axes[len(pairs):]:
            ax.set_visible(False)

        fig.suptitle(
            f"{model_name} – 2-D PDP at epoch {snap.epoch}",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        fname = f"08_2d_pdp_{model_name}_epoch{snap.epoch}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved: {fname}")


# ── 6i  Local explanation evolution (same sample across epochs) ──────────────
def plot_local_evolution(
    snaps: list[Snapshot],
    X_ex_raw: np.ndarray,
    model_label: str,
    sample_idx: int = 0,
) -> None:
    """
    Waterfall plots for the same fixed sample at every SHAP snapshot epoch,
    showing how the model's local reasoning changes over training.
    """
    shap_snaps = [s for s in snaps if s.shap_values is not None]
    n = len(shap_snaps)

    # ── render each waterfall to its own figure, then tile into one image ─────
    # SHAP waterfall manipulates the axes heavily; rendering standalone avoids
    # overlap artefacts that occur in dense subplot grids.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    import io

    cell_images: list[Image.Image] = []
    for snap in shap_snaps:
        expl = shap.Explanation(
            values        = snap.shap_values[sample_idx],
            base_values   = 0.0,
            data          = X_ex_raw[sample_idx],
            feature_names = FEATURE_COLS,
        )
        tmp_fig = plt.figure(figsize=(6, 5))
        shap.plots.waterfall(expl, max_display=8, show=False)
        plt.title(f"Epoch {snap.epoch}", fontsize=11, pad=8)
        plt.tight_layout()
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(tmp_fig)
        buf.seek(0)
        cell_images.append(Image.open(buf).copy())
        buf.close()

    # ── tile individual images into a grid ────────────────────────────────────
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    cw   = max(img.width for img in cell_images)
    ch   = max(img.height for img in cell_images)
    pad  = 20  # pixels between cells

    canvas_w = cols * cw + (cols - 1) * pad
    canvas_h = rows * ch + (rows - 1) * pad + 60  # extra for suptitle
    canvas   = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    for idx, img in enumerate(cell_images):
        r, c = divmod(idx, cols)
        x = c * (cw + pad) + (cw - img.width) // 2
        y = 60 + r * (ch + pad) + (ch - img.height) // 2
        canvas.paste(img, (x, y))

    # ── add suptitle via matplotlib ───────────────────────────────────────────
    title_fig = plt.figure(figsize=(canvas_w / DPI, 0.5))
    title_fig.text(
        0.5, 0.5,
        f"{model_label} – Local explanation evolution (sample {sample_idx})",
        ha="center", va="center", fontsize=14,
    )
    title_buf = io.BytesIO()
    title_fig.savefig(title_buf, format="png", dpi=DPI, bbox_inches="tight",
                      facecolor="white")
    plt.close(title_fig)
    title_buf.seek(0)
    title_img = Image.open(title_buf).copy()
    # centre the title image at the top
    tx = (canvas_w - title_img.width) // 2
    canvas.paste(title_img, (tx, 0))
    title_buf.close()

    fname = f"09_local_evolution_{model_label}.png"
    canvas.save(str(OUTPUT_DIR / fname))
    for img in cell_images:
        img.close()
    title_img.close()
    print(f"  saved: {fname}")


# ── 6j  H-statistic evolution (Friedman) ─────────────────────────────────────
def plot_h_statistic_evolution(
    snaps: list[Snapshot],
    hidden_dims: list[int],
    dropout: float,
    scaler: StandardScaler,
    X_va_df: pd.DataFrame,
    model_label: str = "model",
) -> None:
    """
    Track Friedman's H-statistic for CONTRAST_PAIRS across epochs that have
    saved model_state.  This measures true PDP-based interaction strength,
    complementing the SHAP-correlation proxy in plot_interaction_evolution().
    """
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping H-statistic evolution.")
        return

    pairs = CONTRAST_PAIRS if CONTRAST_PAIRS else [(f1, f2, "signal") for f1, f2 in GT_PAIRS]
    if not pairs:
        print("  [warn] no pairs, skipping H-statistic evolution.")
        return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)
    epochs = [s.epoch for s in proxy_snaps]

    # Compute H-statistic for each pair at each epoch
    pair_h: dict[str, list[float]] = {}
    for f1, f2, cat in pairs:
        key = f"{f1}×{f2} [{cat}]"
        pair_h[key] = []
        i1 = FEATURE_COLS.index(f1)
        i2 = FEATURE_COLS.index(f2)
        for snap in proxy_snaps:
            m = PurchaseNN(len(FEATURE_COLS), hidden_dims, dropout)
            m.load_state_dict(snap.model_state)
            m.eval()
            w = ModelWrapper(m, scaler)
            h = compute_h_statistic(w, X_pdp, i1, i2, grid_res=25)
            pair_h[key].append(h)

    fig, ax = plt.subplots(figsize=(12, 5))
    for (f1, f2, cat), key in zip(pairs, pair_h.keys()):
        style = _PAIR_STYLE.get(cat, dict(color="gray", marker=".", linewidth=1.5, linestyle="-"))
        ax.plot(epochs, pair_h[key], label=key, **style)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Friedman H-statistic", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.set_title(f"{model_label} – H-statistic evolution (PDP-based interaction strength)",
                 fontsize=13)
    plt.tight_layout()
    fname = f"10_h_statistic_evolution_{model_label}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ── 6k  Feature interaction network graphs ───────────────────────────────────
def plot_interaction_network(
    snaps: list[Snapshot],
    model_label: str = "model",
    threshold: float = 0.30,
) -> None:
    """
    Network graph at each INTERACTION_EPOCH:
      - Nodes = features (green = signal, red = noise)
      - Edges = interaction proxy strength (width ∝ magnitude)
      - Suppress edges below *threshold*
    """
    proxy_snaps = [s for s in snaps if s.interaction_proxy is not None]
    if not proxy_snaps:
        print("  [warn] no interaction proxy snapshots, skipping network graph.")
        return

    n_epochs = len(proxy_snaps)
    fig, axes = plt.subplots(1, n_epochs, figsize=(7 * n_epochs, 7))
    if n_epochs == 1:
        axes = [axes]

    signal_set = set(SIGNAL_FEATURES)

    for ax, snap in zip(axes, proxy_snaps):
        M = snap.interaction_proxy
        n_feat = min(M.shape[0], len(FEATURE_COLS))

        G = nx.Graph()
        for i in range(n_feat):
            G.add_node(FEATURE_COLS[i])

        for i in range(n_feat):
            for j in range(i + 1, n_feat):
                strength = (M[i, j] + M[j, i]) / 2  # symmetrise
                if strength >= threshold:
                    G.add_edge(FEATURE_COLS[i], FEATURE_COLS[j], weight=strength)

        pos = nx.spring_layout(G, seed=42, k=1.5)

        # Node colours
        node_colours = [
            "#2ca02c" if f in signal_set else "#d62728"
            for f in G.nodes()
        ]

        # Edge widths & colours
        edges = G.edges(data=True)
        widths = [e[2]["weight"] * 8 for e in edges]
        edge_colours = [plt.cm.Oranges(e[2]["weight"]) for e in edges]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                               node_size=600, edgecolors="black", linewidths=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
        if edges:
            nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                                   edge_color=edge_colours, alpha=0.8)

        ax.set_title(f"Epoch {snap.epoch}", fontsize=12)
        ax.axis("off")

    # ── Legend for node colours ───────────────────────────────────────────────
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#2ca02c", label="Signal feature"),
        mpatches.Patch(color="#d62728", label="Noise feature"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{model_label} – Feature interaction network  (threshold={threshold})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fname = f"11_interaction_network_{model_label}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ── 6l  2-D PDP contour time-lapse (unified colour scale) ───────────────────
def plot_2d_pdp_timelapse(
    model_name: str,
    snaps: list[Snapshot],
    hidden_dims: list[int],
    dropout: float,
    scaler: StandardScaler,
    X_va_df: pd.DataFrame,
) -> None:
    """
    Unified time-lapse grid: rows = feature pairs, columns = epochs.
    All subplots share the same colour scale for honest comparison.
    Uses CONTRAST_PAIRS[:6] to include non-GT pairs that may show crossover.
    """
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping 2-D PDP time-lapse.")
        return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)

    # Use up to 6 contrast pairs (prioritise GT pairs first)
    gt_set = {(a, b) for a, b in GT_PAIRS}
    gt_contrast = [(f1, f2, c) for f1, f2, c in CONTRAST_PAIRS
                   if (f1, f2) in gt_set or (f2, f1) in gt_set]
    non_gt = [(f1, f2, c) for f1, f2, c in CONTRAST_PAIRS
              if (f1, f2) not in gt_set and (f2, f1) not in gt_set]
    pairs = (gt_contrast + non_gt)[:6]
    if not pairs:
        print("  [warn] no pairs for 2-D PDP time-lapse.")
        return

    n_pairs  = len(pairs)
    n_epochs = len(proxy_snaps)

    # First pass: compute all Z arrays & find global vmin/vmax
    Z_all = {}  # (pair_idx, epoch_idx) → (G1, G2, Z)
    global_vmin = np.inf
    global_vmax = -np.inf

    for ei, snap in enumerate(proxy_snaps):
        m = PurchaseNN(len(FEATURE_COLS), hidden_dims, dropout)
        m.load_state_dict(snap.model_state)
        m.eval()
        w = ModelWrapper(m, scaler)
        for pi, (f1, f2, _) in enumerate(pairs):
            i1 = FEATURE_COLS.index(f1)
            i2 = FEATURE_COLS.index(f2)
            pdp_res = partial_dependence(
                w, X_pdp, features=[(i1, i2)],
                grid_resolution=30, kind="average",
            )
            g1, g2 = pdp_res["grid_values"]
            Z = pdp_res["average"][0]
            G1, G2 = np.meshgrid(g1, g2)
            Z_all[(pi, ei)] = (G1, G2, Z)
            global_vmin = min(global_vmin, Z.min())
            global_vmax = max(global_vmax, Z.max())

    # Second pass: plot with shared colour range
    fig, axes = plt.subplots(n_pairs, n_epochs,
                              figsize=(6 * n_epochs, 5 * n_pairs),
                              squeeze=False)

    levels = np.linspace(global_vmin, global_vmax, 20)

    for pi, (f1, f2, cat) in enumerate(pairs):
        for ei, snap in enumerate(proxy_snaps):
            ax = axes[pi][ei]
            G1, G2, Z = Z_all[(pi, ei)]
            cs = ax.contourf(G1, G2, Z.T, levels=levels, cmap="viridis", alpha=0.8)
            ax.set_xlabel(f1, fontsize=9)
            ax.set_ylabel(f2, fontsize=9)
            if pi == 0:
                ax.set_title(f"Epoch {snap.epoch}", fontsize=11, fontweight="bold")
            if ei == 0:
                ax.annotate(f"{f1}×{f2}\n[{cat}]", xy=(-0.35, 0.5),
                            xycoords="axes fraction", fontsize=10,
                            ha="center", va="center", rotation=90)

    # Single shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(cs, cax=cbar_ax, label="Partial dependence")

    fig.suptitle(f"{model_name} – 2-D PDP time-lapse (shared colour scale)",
                 fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    fname = f"12_2d_pdp_timelapse_{model_name}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ── 6m  Faceted ICE plots coloured by conditioning feature ───────────────────
def plot_ice_faceted(
    snaps: list[Snapshot],
    hidden_dims: list[int],
    dropout: float,
    scaler: StandardScaler,
    X_va_df: pd.DataFrame,
    model_label: str = "model",
    n_ice_samples: int = 100,
) -> None:
    """
    Faceted ICE plots: rows = GT feature pairs, columns = epochs.
    Each panel shows individual prediction curves for feature f1,
    coloured by the observation's value of f2.
    Fanning lines → modifier; crossing lines → crossover.
    """
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping faceted ICE plots.")
        return

    pairs = GT_PAIRS[:3]  # keep readable
    if not pairs:
        print("  [warn] no GT_PAIRS, skipping faceted ICE.")
        return

    n_pairs  = len(pairs)
    n_epochs = len(proxy_snaps)
    grid_pts = 50

    fig, axes = plt.subplots(n_pairs, n_epochs,
                              figsize=(6 * n_epochs, 5 * n_pairs),
                              squeeze=False)

    # Sub-sample observations for ICE
    X_ice = X_va_df.sample(min(n_ice_samples, len(X_va_df)), random_state=42)
    X_ice_arr = X_ice.values.astype(np.float32)

    for ei, snap in enumerate(proxy_snaps):
        m = PurchaseNN(len(FEATURE_COLS), hidden_dims, dropout)
        m.load_state_dict(snap.model_state)
        m.eval()
        w = ModelWrapper(m, scaler)

        for pi, (f1, f2) in enumerate(pairs):
            ax = axes[pi][ei]
            i1 = FEATURE_COLS.index(f1)
            i2 = FEATURE_COLS.index(f2)

            f1_vals = X_ice_arr[:, i1]
            f2_vals = X_ice_arr[:, i2]
            grid = np.linspace(
                np.percentile(f1_vals, 2),
                np.percentile(f1_vals, 98),
                grid_pts,
            )

            # Normalise f2 for colouring
            f2_min, f2_max = f2_vals.min(), f2_vals.max()
            f2_range = f2_max - f2_min if (f2_max - f2_min) > 1e-12 else 1.0
            f2_norm = (f2_vals - f2_min) / f2_range

            cmap = plt.cm.coolwarm

            for obs_idx in range(len(X_ice_arr)):
                row_tiled = np.tile(X_ice_arr[obs_idx], (grid_pts, 1))
                row_tiled[:, i1] = grid
                preds = w.predict_proba(row_tiled)[:, 1]
                ax.plot(grid, preds, color=cmap(f2_norm[obs_idx]),
                        alpha=0.3, linewidth=0.6)

            ax.set_xlabel(f1, fontsize=9)
            ax.set_ylabel("P(y=1)", fontsize=9)
            if pi == 0:
                ax.set_title(f"Epoch {snap.epoch}", fontsize=11, fontweight="bold")
            if ei == 0:
                ax.annotate(f"{f1}×{f2}", xy=(-0.35, 0.5),
                            xycoords="axes fraction", fontsize=10,
                            ha="center", va="center", rotation=90)

    # Shared colourbar for f2
    import matplotlib.cm as mcm
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Conditioning feature value (normalised)")

    fig.suptitle(f"{model_label} – ICE plots (colour = conditioning feature)",
                 fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    fname = f"13_ice_faceted_{model_label}.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    ph(f"Loading synth.functions[{SYNTH_FUNC_IDX}]")
    (X_tr, y_tr, X_va, y_va, X_te, y_te,
     X_va_raw, X_tr_raw, scaler, X_va_df, ground_truth) = load_synth_data()

    n_features = X_tr.shape[1]
    print(f"  Train={len(X_tr)}  Valid={len(X_va)}  Test={len(X_te)}")
    print(f"  Features={n_features}  Signal={SIGNAL_FEATURES}  Noise={NOISE_FEATURES}")
    print(f"  GT pairwise:  {ground_truth['pairwise']}")
    print(f"  GT any-order: {ground_truth['any_order']}")
    print(f"  GT_PAIRS (name): {GT_PAIRS}")

    # ── Fixed SHAP background & explain sets (raw, unscaled) ──────────────────
    rng    = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_tr_raw), size=min(N_SHAP_BG, len(X_tr_raw)), replace=False)
    ex_idx = rng.choice(len(X_va_raw), size=min(N_SHAP_EXPLAIN, len(X_va_raw)), replace=False)
    X_bg     = X_tr_raw[bg_idx]
    X_ex_raw = X_va_raw[ex_idx]

    # ── Train single model ────────────────────────────────────────────────────
    wrapper, snaps = train_model(
        n_features   = n_features,
        hidden_dims  = HIDDEN_DIMS,
        dropout      = DROPOUT,
        weight_decay = WEIGHT_DECAY,
        X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
        X_bg=X_bg, X_ex=X_ex_raw, scaler=scaler,
        model_name=f"synth_F{SYNTH_FUNC_IDX}",
    )

    # ── Final AUC summary ─────────────────────────────────────────────────────
    ph("Final metrics")
    last = snaps[-1]
    print(f"  train_auc={last.train_auc:.4f}  valid_auc={last.valid_auc:.4f}")

    # ── Visualisations ────────────────────────────────────────────────────────
    label = f"synth_F{SYNTH_FUNC_IDX}"
    ph("Plotting")

    # 1. Training curves
    plot_training_curves(snaps, label)

    # 2. Feature importance evolution
    plot_importance_evolution(snaps, label)

    # 3. Noise inflation
    plot_noise_inflation(snaps, label)

    # 4. Beeswarms at early/optimal/late
    plot_beeswarms(snaps, X_ex_raw, label)

    # 5. Interaction proxy heatmaps
    plot_interaction_proxy_heatmaps(snaps, label)

    # 6. Interaction evolution line plot
    plot_interaction_evolution(snaps, X_ex_raw, label)

    # 7. Interaction type classification + conditional PDPs
    plot_interaction_classification(wrapper, snaps, X_ex_raw, X_va_df, label)

    # 8. 2-D PDP contour at interaction epochs
    plot_2d_pdp_at_epochs(label, snaps, HIDDEN_DIMS, DROPOUT, scaler, X_va_df)

    # 9. Local explanation evolution
    plot_local_evolution(snaps, X_ex_raw, label)

    # 10. H-statistic evolution (Friedman)
    plot_h_statistic_evolution(snaps, HIDDEN_DIMS, DROPOUT, scaler, X_va_df, label)

    # 11. Feature interaction network graphs
    plot_interaction_network(snaps, label)

    # 12. 2-D PDP time-lapse (unified colour scale)
    plot_2d_pdp_timelapse(label, snaps, HIDDEN_DIMS, DROPOUT, scaler, X_va_df)

    # 13. Faceted ICE plots
    plot_ice_faceted(snaps, HIDDEN_DIMS, DROPOUT, scaler, X_va_df, label)

    ph("All outputs saved")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()