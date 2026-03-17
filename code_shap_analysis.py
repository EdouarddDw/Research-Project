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

import networkx as nx
from multilayer_perceptron import MLP, train as mlp_train

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

OUTPUT_DIR = BASE_DIR / "shap_outputs2"

# ── Config ────────────────────────────────────────────────────────────────────
ALL_FUNC_INDICES  = list(range(len(synth.functions)))   # F1 … F10
NUM_SAMPLES       = 500
SEED              = 42

SNAPSHOT_EPOCHS    = [1, 5, 10, 15, 25, 35, 50]
INTERACTION_EPOCHS = [5, 15, 25, 35, 50]
TOTAL_EPOCHS       = 50
BATCH_SIZE         = 128
LR                 = 0.01
N_SHAP_BG          = 80
N_SHAP_EXPLAIN     = 60
PDP_SUBSAMPLE      = 50
DPI                = 130

HIDDEN_DIMS  = [128, 64, 32]

# Two experimental conditions for Goal 2
EXPERIMENT_CONDITIONS = [
    ("no_reg",  0.0,  0.0),    # unregularized — should overfit
    ("reg",     0.3,  1e-4),   # regularized   — should stay generalised
]
N_NOISE_COLS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Runtime globals (reset per function) ──────────────────────────────────────
FEATURE_COLS:    list[str] = []
SIGNAL_FEATURES: list[str] = []
NOISE_FEATURES:  list[str] = []
GT_PAIRS:        list[tuple[str, str]] = []
CONTRAST_PAIRS:  list[tuple[str, str, str]] = []

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

class ModelWrapper(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, model: MLP = None, scaler: StandardScaler = None) -> None:
        self.model  = model
        self.scaler = scaler
        self.classes_   = np.array([0, 1])
        self.is_fitted_ = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xs = self.scaler.transform(X)
        t  = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = self.model(t).cpu().numpy().squeeze()
        proba1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1 - proba1, proba1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

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
    shap_values:        Optional[np.ndarray] = None
    interaction_proxy:  Optional[np.ndarray] = None
    model_state:        Optional[dict]       = None


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _derive_feature_metadata(ground_truth: dict, n_features: int) -> None:
    global FEATURE_COLS, SIGNAL_FEATURES, NOISE_FEATURES, GT_PAIRS, CONTRAST_PAIRS

    FEATURE_COLS = [f"x{i}" for i in range(n_features)]

    signal_idx = sorted({i for tup in ground_truth["pairwise"] + ground_truth["any_order"]
                         for i in tup})
    noise_idx  = sorted(set(range(n_features)) - set(signal_idx))

    SIGNAL_FEATURES = [FEATURE_COLS[i] for i in signal_idx]
    NOISE_FEATURES  = [FEATURE_COLS[i] for i in noise_idx]

    GT_PAIRS = [(FEATURE_COLS[a], FEATURE_COLS[b])
                for (a, b) in ground_truth["pairwise"]]

    gt_set = {(a, b) for a, b in GT_PAIRS}

    CONTRAST_PAIRS = []
    for f1, f2 in combinations(SIGNAL_FEATURES, 2):
        CONTRAST_PAIRS.append((f1, f2, "signal"))

    mixed_count = 0
    for sf in SIGNAL_FEATURES:
        for nf in NOISE_FEATURES:
            CONTRAST_PAIRS.append((sf, nf, "mixed"))
            mixed_count += 1
            if mixed_count >= 4:
                break
        if mixed_count >= 4:
            break

    noise_combos = list(combinations(NOISE_FEATURES, 2))[:2]
    CONTRAST_PAIRS += [(f1, f2, "noise") for f1, f2 in noise_combos]


def load_synth_data(
    func_idx:    int = 0,          # ← plain int, not a list
    num_samples: int = NUM_SAMPLES,
    seed:        int = SEED,
    valid_frac:  float = 0.2,
    test_frac:   float = 0.2,
):
    X, Y_cont, interactions_raw = synth.functions[func_idx](
        num_samples=num_samples, seed=seed
    )
    Y = (Y_cont > np.median(Y_cont)).astype(np.float32)

    n          = len(X)
    n_features = X.shape[1]

    # ── Normalise new synth format → ground_truth dict ───────────────────────
    # New synth returns a list of sets with 1-based indices e.g. [{1,2},{3,4,5}]
    # Convert to 0-based tuples and build the pairwise / any_order dict that
    # _derive_feature_metadata expects.
    interactions_0 = [tuple(sorted(i - 1 for i in s)) for s in interactions_raw]
    pairwise = [t for t in interactions_0 if len(t) == 2]
    for group in interactions_0:
        if len(group) > 2:
            for pair in combinations(group, 2):
                if pair not in pairwise:
                    pairwise.append(pair)
    ground_truth = {"pairwise": pairwise, "any_order": interactions_0}

    _derive_feature_metadata(ground_truth, n_features)

    # Append pure-noise columns so NOISE_FEATURES is always non-empty (Goal 2)
    if N_NOISE_COLS > 0:
        rng_noise = np.random.default_rng(seed + 777)
        X_noise   = rng_noise.uniform(-1, 1, (n, N_NOISE_COLS)).astype(np.float32)
        noise_names = [f"noise{i}" for i in range(N_NOISE_COLS)]
        X = np.hstack([X, X_noise])
        NOISE_FEATURES.extend(noise_names)
        FEATURE_COLS.extend(noise_names)
        n_features = X.shape[1]

    valid_size = int(n * valid_frac)
    test_size  = int(n * test_frac)
    n_train    = n - valid_size - test_size
    assert n_train > 0, (
        f"n_train={n_train} is not positive. "
        f"num_samples={num_samples}, valid_frac={valid_frac}, test_frac={test_frac}"
    )
    X_tr_raw = X[:n_train].astype(np.float32)
    X_va_raw = X[n_train:n_train + valid_size].astype(np.float32)
    X_te_raw = X[n_train + valid_size:].astype(np.float32)

    y_tr = Y[:n_train].copy()
    y_va = Y[n_train:n_train + valid_size]
    y_te = Y[n_train + valid_size:]

    # Inject 15% label noise into training set only so the model has
    # something to memorise while val/test remain clean ground-truth.
    y_tr_clean = y_tr.copy()   # kept for honest train-AUC evaluation
    rng        = np.random.default_rng(seed)
    noise_mask = rng.random(len(y_tr)) < 0.15
    y_tr[noise_mask] = 1.0 - y_tr[noise_mask]

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_va   = scaler.transform(X_va_raw).astype(np.float32)
    X_te   = scaler.transform(X_te_raw).astype(np.float32)

    X_va_df = pd.DataFrame(X_va_raw, columns=FEATURE_COLS)

    return (X_tr, y_tr, y_tr_clean, X_va, y_va, X_te, y_te,
            X_va_raw, X_tr_raw, scaler, X_va_df, ground_truth)


def make_loader(X, y, shuffle=True):
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def compute_interaction_proxy(shap_vals: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
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
    try:
        pdp_j  = partial_dependence(wrapper, X_pdp, features=[f1_idx],
                                    grid_resolution=grid_res, kind="average")
        pdp_k  = partial_dependence(wrapper, X_pdp, features=[f2_idx],
                                    grid_resolution=grid_res, kind="average")
        pdp_jk = partial_dependence(wrapper, X_pdp, features=[(f1_idx, f2_idx)],
                                    grid_resolution=grid_res, kind="average")

        pdp_j_vals  = pdp_j["average"][0]
        pdp_k_vals  = pdp_k["average"][0]
        pdp_jk_vals = pdp_jk["average"][0]

        f_bar    = pdp_jk_vals.mean()
        pdp_j_c  = pdp_j_vals  - f_bar
        pdp_k_c  = pdp_k_vals  - f_bar
        pdp_jk_c = pdp_jk_vals - f_bar

        residual = pdp_jk_c - pdp_j_c[:, None] - pdp_k_c[None, :]
        numer    = np.sum(residual ** 2)
        denom    = np.sum(pdp_jk_c ** 2)
        if denom < 1e-20:
            return 0.0
        return float(np.sqrt(np.clip(numer / denom, 0, 1)))
    except Exception:
        return 0.0


def train_model(
    n_features: int,
    hidden_dims: list[int],
    dropout: float,
    weight_decay: float,
    X_tr, y_tr, y_tr_clean, X_va, y_va,
    X_bg, X_ex,
    scaler: StandardScaler,
    model_name: str = "model",
) -> tuple[ModelWrapper, list[Snapshot]]:
    ph(f"Training  {model_name}  |  layers={hidden_dims}  wd={weight_decay}")

    # ── Build model & data loaders ────────────────────────────────────────────
    model = MLP(
        num_features=n_features,
        hidden_units=hidden_dims,
        dropout=dropout,
        use_main_effect_nets=False,
    ).to(DEVICE)

    # multilayer_perceptron.train expects DataLoaders with (inputs, labels)
    # labels must be float and shape (N, 1) for MSE / BCE
    def _make_loader(X, y, shuffle=True):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    data_loaders = {
        "train": _make_loader(X_tr, y_tr, shuffle=True),
        "val":   _make_loader(X_va, y_va, shuffle=False),
    }

    # ── Train via multilayer_perceptron.train ─────────────────────────────────
    trained_net, test_loss, snap_state_dicts = mlp_train(
        net          = model,
        data_loaders = data_loaders,
        criterion    = nn.BCEWithLogitsLoss(),
        nepochs      = TOTAL_EPOCHS,
        verbose      = True,
        early_stopping = False,
        l2_const     = weight_decay,
        learning_rate = LR,
        opt_func     = torch.optim.Adam,
        device       = DEVICE,
        save_snapshots  = True,
        snapshot_epochs = SNAPSHOT_EPOCHS,
        snapshot_dir    = str(OUTPUT_DIR / "snapshots" / model_name),
    )

    wrapper = ModelWrapper(trained_net, scaler)

    # ── Build Snapshot objects from saved state dicts ─────────────────────────
    criterion = nn.BCEWithLogitsLoss()

    def _eval_auc_loss(state_dict, X_scaled, y):
        """Reload weights, run inference, return (loss, auc)."""
        m = MLP(num_features=n_features, hidden_units=hidden_dims, dropout=dropout).to(DEVICE)
        m.load_state_dict(state_dict)
        m.eval()
        with torch.no_grad():
            logits = m(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy().squeeze()
        loss  = criterion(torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)).item()
        proba = 1 / (1 + np.exp(-logits))
        auc   = roc_auc_score(y, proba)
        return loss, auc

    snapshots: list[Snapshot] = []

    for epoch, state_dict in sorted(snap_state_dicts.items()):
        print(f"  Post-processing epoch {epoch:3d} …", end=" ", flush=True)

        tr_loss, tr_auc = _eval_auc_loss(state_dict, X_tr, y_tr_clean)
        va_loss, va_auc = _eval_auc_loss(state_dict, X_va, y_va)

        snap = Snapshot(epoch, tr_loss, va_loss, tr_auc, va_auc)

        # ── SHAP ──────────────────────────────────────────────────────────────
        tmp_wrapper = ModelWrapper(
            MLP(num_features=n_features, hidden_units=hidden_dims, dropout=dropout).to(DEVICE),
            scaler,
        )
        tmp_wrapper.model.load_state_dict(state_dict)
        tmp_wrapper.model.eval()

        explainer = shap.Explainer(lambda x: tmp_wrapper.predict_proba(x)[:, 1],
                                masker=shap.maskers.Independent(X_bg))
        expl = explainer(X_ex, max_evals=200)          # tune max_evals for speed/quality
        # `expl.values` has shape (n_samples, n_features)
        snap.shap_values = np.asarray(expl.values)
        print(f"SHAP done", end=" ")

        # ── Interaction proxy at INTERACTION_EPOCHS ───────────────────────────
        if epoch in INTERACTION_EPOCHS:
            snap.interaction_proxy = compute_interaction_proxy(snap.shap_values, X_ex)
            snap.model_state       = {k: v.cpu().clone() for k, v in state_dict.items()}
            print(f"+ proxy", end=" ")

        print(f"[{_elapsed()}]")
        snapshots.append(snap)

    return wrapper, snapshots


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INTERACTION TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_interaction(shap_vals, X_raw, f1, f2) -> str:
    i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)
    shap_f1 = shap_vals[:, i1]
    x_f1, x_f2 = X_raw[:, i1], X_raw[:, i2]

    median_f2 = np.median(x_f2)
    low_mask  = x_f2 <= median_f2
    high_mask = x_f2 > median_f2

    if low_mask.sum() < 5 or high_mask.sum() < 5:
        return "insufficient_data"

    def _slope(x, y):
        if len(x) < 3 or np.std(x) < 1e-12:
            return 0.0
        return np.polyfit(x, y, 1)[0]

    slope_low  = _slope(x_f1[low_mask],  shap_f1[low_mask])
    slope_high = _slope(x_f1[high_mask], shap_f1[high_mask])

    if slope_low * slope_high < 0:
        return "crossover"

    abs_lo = abs(slope_low)
    abs_hi = abs(slope_high)
    ratio  = max(abs_lo, abs_hi) / max(min(abs_lo, abs_hi), 1e-12)

    if ratio > 2.0:
        return "modifier"
    if ratio < 1.2:
        return "additive"
    return "symmetric"


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS  (all accept func_label for namespaced output filenames)
# ══════════════════════════════════════════════════════════════════════════════

_PAIR_STYLE = {
    "signal": dict(color="#2ca02c", marker="o", linewidth=2.5, linestyle="-"),
    "mixed":  dict(color="#ff7f0e", marker="s", linewidth=2.0, linestyle="--"),
    "noise":  dict(color="#d62728", marker="^", linewidth=2.0, linestyle=":"),
}


def _fname(output_dir: Path, num: str, desc: str, func_label: str) -> Path:
    """Return namespaced output path, e.g. shap_outputs/F3/01_training_curves_F3.png"""
    d = output_dir / func_label
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{num}_{desc}_{func_label}.png"


def plot_training_curves(snaps, func_label, output_dir):
    epochs = [s.epoch for s in snaps]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, [s.train_loss for s in snaps], "b-",  label="train")
    axes[0].plot(epochs, [s.valid_loss for s in snaps], "b--", label="valid")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
    axes[0].set_title(f"{func_label} – Training vs Validation Loss"); axes[0].legend()
    axes[1].plot(epochs, [s.train_auc for s in snaps], "b-",  label="train")
    axes[1].plot(epochs, [s.valid_auc for s in snaps], "b--", label="valid")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    axes[1].set_title(f"{func_label} – Training vs Validation AUC"); axes[1].legend()
    plt.tight_layout()
    p = _fname(output_dir, "01", "training_curves", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_importance_evolution(snaps, func_label, output_dir):
    shap_snaps = [s for s in snaps if s.shap_values is not None]
    epochs = [s.epoch for s in shap_snaps]
    mat = np.array([np.abs(s.shap_values).mean(axis=0) for s in shap_snaps]).T
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pd.DataFrame(mat, index=FEATURE_COLS, columns=epochs),
                cmap="YlOrRd", ax=ax, linewidths=0.3)
    ax.set_title(f"{func_label} – Feature importance evolution (mean |SHAP|)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Feature")
    plt.tight_layout()
    p = _fname(output_dir, "02", "importance_evolution", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_noise_inflation(snaps, func_label, output_dir):
    def _mean_shap_group(snaps, cols):
        result = []
        for s in snaps:
            if s.shap_values is not None:
                idx = [FEATURE_COLS.index(c) for c in cols]
                result.append((s.epoch, np.abs(s.shap_values[:, idx]).mean()))
        return zip(*result) if result else ([], [])

    fig, ax = plt.subplots(figsize=(10, 5))
    sig_e, sig_v = _mean_shap_group(snaps, SIGNAL_FEATURES)
    noi_e, noi_v = _mean_shap_group(snaps, NOISE_FEATURES)
    ax.plot(list(sig_e), list(sig_v), color="steelblue", lw=2, label="Signal features")
    if NOISE_FEATURES:
        ax.plot(list(noi_e), list(noi_v), color="tomato", lw=2,
                linestyle="--", label="Noise features")
    ax.set_title(f"{func_label} – Signal vs Noise SHAP")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mean |SHAP|"); ax.legend()
    plt.tight_layout()
    p = _fname(output_dir, "03", "noise_inflation", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_beeswarms(snaps, X_ex_raw, func_label, output_dir):
    from PIL import Image
    import io

    shap_snaps   = [s for s in snaps if s.shap_values is not None]
    target_epochs = [SNAPSHOT_EPOCHS[0], 50, SNAPSHOT_EPOCHS[-1]]
    chosen = [min(shap_snaps, key=lambda s: abs(s.epoch - te)) for te in target_epochs]
    labels = ["Early", "Optimal", "Late (overfit)"]

    cell_images = []
    for snap, lbl in zip(chosen, labels):
        expl = shap.Explanation(values=snap.shap_values, data=X_ex_raw,
                                feature_names=FEATURE_COLS)
        tmp_fig = plt.figure(figsize=(7, 6))
        shap.plots.beeswarm(expl, max_display=12, show=False)
        plt.title(f"{func_label} – {lbl} (epoch {snap.epoch})", fontsize=11, pad=8)
        plt.tight_layout()
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(tmp_fig); buf.seek(0)
        cell_images.append(Image.open(buf).copy()); buf.close()

    pad  = 20
    cw   = max(img.width  for img in cell_images)
    ch   = max(img.height for img in cell_images)
    canvas = Image.new("RGB", (len(cell_images) * cw + (len(cell_images)-1)*pad, ch),
                       (255, 255, 255))
    for idx, img in enumerate(cell_images):
        canvas.paste(img, (idx*(cw+pad) + (cw-img.width)//2, (ch-img.height)//2))

    p = _fname(output_dir, "04", "beeswarms", func_label)
    canvas.save(str(p))
    for img in cell_images: img.close()
    print(f"  saved: {p.name}")


def plot_interaction_proxy_heatmaps(snaps, func_label, output_dir):
    proxy_snaps = [s for s in snaps if s.interaction_proxy is not None]
    n = len(proxy_snaps)
    if n == 0:
        print("  [warn] no interaction proxy snapshots, skipping."); return

    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    n_show = min(len(FEATURE_COLS), 10)
    for ax, snap in zip(axes, proxy_snaps):
        df = pd.DataFrame(snap.interaction_proxy[:n_show, :n_show],
                          index=FEATURE_COLS[:n_show], columns=FEATURE_COLS[:n_show])
        sns.heatmap(df, cmap="coolwarm", vmin=0, vmax=1, ax=ax,
                    linewidths=0.3, square=True)
        ax.set_title(f"Epoch {snap.epoch}\n|corr(SHAP_i, X_j)|")
    fig.suptitle(f"{func_label} – Interaction proxy evolution", fontsize=13, y=1.02)
    plt.tight_layout()
    p = _fname(output_dir, "05", "interaction_proxy", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_interaction_evolution(snaps, X_ex_raw, func_label, output_dir):
    if not CONTRAST_PAIRS:
        print("  [warn] no CONTRAST_PAIRS, skipping."); return

    shap_snaps = [s for s in snaps if s.shap_values is not None]
    if not shap_snaps:
        print("  [warn] no SHAP snapshots, skipping."); return

    epochs = [s.epoch for s in shap_snaps]
    fig, ax = plt.subplots(figsize=(14, 6))
    per_cat: dict[str, list[np.ndarray]] = {"signal": [], "mixed": [], "noise": []}

    for f1, f2, cat in CONTRAST_PAIRS:
        i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)
        proxies = []
        for s in shap_snaps:
            c = np.corrcoef(s.shap_values[:, i1], X_ex_raw[:, i2])[0, 1]
            proxies.append(abs(c) if np.isfinite(c) else 0.0)
        proxies = (pd.Series(proxies)
                   .rolling(window=3, min_periods=1, center=True)
                   .mean().to_numpy())
        style = _PAIR_STYLE[cat]
        ax.plot(epochs, proxies, color=style["color"], linewidth=1.0,
                alpha=0.25, linestyle=style.get("linestyle", "-"))
        per_cat[cat].append(proxies)

    for cat, curves in per_cat.items():
        if not curves: continue
        mean_curve = np.mean(np.vstack(curves), axis=0)
        style = _PAIR_STYLE[cat]
        ax.plot(epochs, mean_curve, label=f"{cat} (mean)", color=style["color"],
                linewidth=3.0, linestyle=style.get("linestyle", "-"))

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("|corr(SHAP_i, X_j)|  (interaction proxy)", fontsize=11)
    ax.set_ylim(0, 0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    ax.set_title(f"{func_label} – Interaction proxy evolution", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    p = _fname(output_dir, "06", "interaction_evolution", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_interaction_classification(wrapper, snaps, X_ex_raw, X_va_df, func_label, output_dir):
    last_shap = next((s for s in reversed(snaps) if s.shap_values is not None), None)
    if last_shap is None:
        print("  [warn] no SHAP snapshots, skipping."); return

    pairs = CONTRAST_PAIRS if CONTRAST_PAIRS else [(f1, f2, "signal") for f1, f2 in GT_PAIRS]
    if not pairs:
        print("  [warn] no pairs, skipping."); return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 5*len(pairs)), squeeze=False)
    cat_colours = {"signal": "#2ca02c", "mixed": "#ff7f0e", "noise": "#d62728"}

    for row, (f1, f2, cat) in enumerate(pairs):
        i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)

        ax_l = axes[row][0]
        sc   = ax_l.scatter(X_ex_raw[:, i1], last_shap.shap_values[:, i1],
                            c=X_ex_raw[:, i2], cmap="coolwarm",
                            alpha=0.5, s=15, edgecolors="none")
        fig.colorbar(sc, ax=ax_l, label=f2)
        ax_l.axhline(0, color="k", lw=0.8, linestyle="--")
        ax_l.set_xlabel(f1); ax_l.set_ylabel(f"SHAP({f1})")
        ax_l.set_title(f"[{cat.upper()}]  {f1} (colour = {f2})",
                       color=cat_colours.get(cat, "k"))

        ax_r    = axes[row][1]
        median  = X_pdp[f2].median()
        X_low   = X_pdp[X_pdp[f2] <= median]
        X_high  = X_pdp[X_pdp[f2] > median]
        f1_idx  = FEATURE_COLS.index(f1)
        full_col = X_pdp.iloc[:, f1_idx].values
        shared_grid = np.linspace(np.percentile(full_col, 2), np.percentile(full_col, 98), 50)

        for subset, colour, lbl in [(X_low, "steelblue", f"{f2} low"),
                                    (X_high, "tomato",   f"{f2} high")]:
            if len(subset) > 10:
                tile = np.tile(subset.values, (len(shared_grid), 1, 1))
                tile[:, :, f1_idx] = shared_grid[:, None]
                G, N, F = tile.shape
                flat = tile.reshape(G*N, F)
                proba = wrapper.predict_proba(flat)[:, 1].reshape(G, N)
                ax_r.plot(shared_grid, proba.mean(axis=1), color=colour,
                          linewidth=2, label=lbl)

        ax_r.set_xlabel(f1); ax_r.set_ylabel("Partial dependence")
        ax_r.set_title(f"PDP of {f1} | {f2} split  [{cat.upper()}]",
                       color=cat_colours.get(cat, "k"))
        ax_r.legend()

    fig.suptitle(f"{func_label} – Interaction classification (classification disabled)", fontsize=13, y=1.01)
    plt.tight_layout()
    p = _fname(output_dir, "07", "interaction_classification", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_2d_pdp_at_epochs(func_label, snaps, hidden_dims, dropout, scaler,
                           X_va_df, output_dir):
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping 2-D PDP at epochs."); return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)
    pairs = GT_PAIRS[:4]
    if not pairs: return

    for snap in proxy_snaps:
        m = MLP(num_features=len(FEATURE_COLS), hidden_units=hidden_dims, dropout=dropout).to(DEVICE)
        m.load_state_dict(snap.model_state); m.eval()
        w = ModelWrapper(m, scaler)

        cols = 2
        rows = int(np.ceil(len(pairs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(9*cols, 8*rows))
        axes = np.array(axes).flatten()

        for ax, (f1, f2) in zip(axes, pairs):
            i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)
            pdp_res = partial_dependence(w, X_pdp, features=[(i1, i2)],
                                         grid_resolution=30, kind="average")
            g1, g2 = pdp_res["grid_values"]
            Z = pdp_res["average"][0]
            G1, G2 = np.meshgrid(g1, g2)
            cs = ax.contourf(G1, G2, Z.T, levels=20, cmap="viridis", alpha=0.8)
            fig.colorbar(cs, ax=ax)
            ax.set_xlabel(f1); ax.set_ylabel(f2); ax.set_title(f"{f1} × {f2}")

        for ax in axes[len(pairs):]: ax.set_visible(False)
        fig.suptitle(f"{func_label} – 2-D PDP at epoch {snap.epoch}", fontsize=13, y=1.01)
        plt.tight_layout()
        p = _fname(output_dir, "08", f"2d_pdp_epoch{snap.epoch}", func_label)
        plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
        print(f"  saved: {p.name}")


def plot_local_evolution(snaps, X_ex_raw, func_label, output_dir, sample_idx=0):
    from PIL import Image
    import io

    shap_snaps = [s for s in snaps if s.shap_values is not None]
    n = len(shap_snaps)
    cell_images = []

    for snap in shap_snaps:
        expl = shap.Explanation(values=snap.shap_values[sample_idx], base_values=0.0,
                                data=X_ex_raw[sample_idx], feature_names=FEATURE_COLS)
        tmp_fig = plt.figure(figsize=(6, 5))
        shap.plots.waterfall(expl, max_display=8, show=False)
        plt.title(f"Epoch {snap.epoch}", fontsize=11, pad=8)
        plt.tight_layout()
        buf = io.BytesIO()
        tmp_fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(tmp_fig); buf.seek(0)
        cell_images.append(Image.open(buf).copy()); buf.close()

    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    cw, ch = max(i.width for i in cell_images), max(i.height for i in cell_images)
    pad = 20
    canvas = Image.new("RGB",
                       (cols*cw + (cols-1)*pad, rows*ch + (rows-1)*pad + 60),
                       (255, 255, 255))
    for idx, img in enumerate(cell_images):
        r, c = divmod(idx, cols)
        canvas.paste(img, (c*(cw+pad)+(cw-img.width)//2,
                           60 + r*(ch+pad) + (ch-img.height)//2))

    title_fig = plt.figure(figsize=(canvas.width/DPI, 0.5))
    title_fig.text(0.5, 0.5,
                   f"{func_label} – Local explanation evolution (sample {sample_idx})",
                   ha="center", va="center", fontsize=14)
    title_buf = io.BytesIO()
    title_fig.savefig(title_buf, format="png", dpi=DPI, bbox_inches="tight",
                      facecolor="white")
    plt.close(title_fig); title_buf.seek(0)
    title_img = Image.open(title_buf).copy()
    canvas.paste(title_img, ((canvas.width - title_img.width)//2, 0))
    title_buf.close()

    p = _fname(output_dir, "09", "local_evolution", func_label)
    canvas.save(str(p))
    for img in cell_images: img.close()
    title_img.close()
    print(f"  saved: {p.name}")


def plot_h_statistic_evolution(snaps, hidden_dims, dropout, scaler,
                                X_va_df, func_label, output_dir):
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping H-statistic evolution."); return

    pairs = CONTRAST_PAIRS if CONTRAST_PAIRS else [(f1, f2, "signal") for f1, f2 in GT_PAIRS]
    if not pairs:
        print("  [warn] no pairs, skipping."); return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)
    epochs = [s.epoch for s in proxy_snaps]
    
    # Step 1: Group data by category
    per_cat_h = {"signal": [], "mixed": [], "noise": []}
    
    for f1, f2, cat in pairs:
        key = f"{f1}×{f2} [{cat}]"
        i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)
        h_vals_for_pair = []
        for snap in proxy_snaps:
            m = MLP(num_features=len(FEATURE_COLS), hidden_units=hidden_dims, dropout=dropout).to(DEVICE)
            m.load_state_dict(snap.model_state); m.eval()
            w = ModelWrapper(m, scaler)
            h_vals_for_pair.append(compute_h_statistic(w, X_pdp, i1, i2, grid_res=10))
        
        per_cat_h[cat].append(np.array(h_vals_for_pair))

    # Step 2: Create a single plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Step 3: Define mean styles based on image_1.png
    _MEAN_STYLE = {
        "signal": dict(color="#2ca02c", linewidth=4.0, linestyle="-"),
        "mixed":  dict(color="#ff7f0e", linewidth=4.0, linestyle="--"),
        "noise":  dict(color="#d62728", linewidth=4.0, linestyle=":"),
    }
    
    # A single color-based style map for individual thin lines
    group_colors = {
        "signal": "#2ca02c",
        "mixed":  "#ff7f0e",
        "noise":  "#d62728",
    }
    
    legend_handles = []

    # Step 4: Plot individual thin lines (no label, with transparency)
    for cat, curves in per_cat_h.items():
        if not curves: continue
        color = group_colors.get(cat, "gray")
        for curve in curves:
            # Thin line, with some alpha to match target image
            ax.plot(epochs, curve, color=color, linewidth=1.0, alpha=0.3)
    
    # Step 5: Plot mean thick lines (with label for legend)
    for cat, curves in per_cat_h.items():
        if not curves: continue
        # Stack curves and calculate the mean along the epoch axis
        mean_curve = np.mean(np.vstack(curves), axis=0)
        mean_style = _MEAN_STYLE[cat]
        # Plot the thick, bold mean line
        line = ax.plot(epochs, mean_curve, label=f"{cat} (mean)", **mean_style)[0]
        legend_handles.append(line)

    # Step 6: Labels, legend, and layout
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Friedman H-statistic", fontsize=11)
    ax.set_ylim(0, 1)
    
    # Create the single, grouped legend from target image style
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    
    ax.set_title(f"{func_label} – H-statistic evolution (grouped)", fontsize=13)
    
    # Adjusted rect to account for the single grouped legend on the right
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Use a namespaced output filename for the grouped plot
    p = _fname(output_dir, "10", "h_statistic_evolution_grouped", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_interaction_network(snaps, func_label, output_dir, threshold=0.10):
    proxy_snaps = [s for s in snaps if s.interaction_proxy is not None]
    if not proxy_snaps:
        print("  [warn] no interaction proxy snapshots, skipping."); return

    n_epochs = len(proxy_snaps)
    fig, axes = plt.subplots(1, n_epochs, figsize=(7*n_epochs, 7))
    if n_epochs == 1: axes = [axes]
    signal_set = set(SIGNAL_FEATURES)

    for ax, snap in zip(axes, proxy_snaps):
        M = snap.interaction_proxy
        n_feat = min(M.shape[0], len(FEATURE_COLS))
        G = nx.Graph()
        for i in range(n_feat):
            G.add_node(FEATURE_COLS[i])
        for i in range(n_feat):
            for j in range(i+1, n_feat):
                strength = (M[i, j] + M[j, i]) / 2
                if strength >= threshold:
                    G.add_edge(FEATURE_COLS[i], FEATURE_COLS[j], weight=strength)

        pos          = nx.spring_layout(G, seed=42, k=1.5)
        node_colours = ["#2ca02c" if f in signal_set else "#d62728" for f in G.nodes()]
        edges        = list(G.edges(data=True))
        widths       = [e[2]["weight"]*8 for e in edges]
        edge_colours = [plt.cm.Oranges(e[2]["weight"]) for e in edges]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours,
                               node_size=600, edgecolors="black", linewidths=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
        if edges:
            nx.draw_networkx_edges(G, pos, ax=ax, width=widths,
                                   edge_color=edge_colours, alpha=0.8)
        ax.set_title(f"Epoch {snap.epoch}", fontsize=12)
        ax.axis("off")

    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color="#2ca02c", label="Signal feature"),
                         mpatches.Patch(color="#d62728", label="Noise feature")],
               loc="lower center", ncol=2, fontsize=11, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{func_label} – Feature interaction network (threshold={threshold})",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p = _fname(output_dir, "11", "interaction_network", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_2d_pdp_timelapse(func_label, snaps, hidden_dims, dropout, scaler,
                           X_va_df, output_dir):
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping 2-D PDP time-lapse."); return

    X_pdp = X_va_df.sample(min(PDP_SUBSAMPLE, len(X_va_df)), random_state=42)
    gt_set      = {(a, b) for a, b in GT_PAIRS}
    gt_contrast = [(f1, f2, c) for f1, f2, c in CONTRAST_PAIRS
                   if (f1, f2) in gt_set or (f2, f1) in gt_set]
    non_gt      = [(f1, f2, c) for f1, f2, c in CONTRAST_PAIRS
                   if (f1, f2) not in gt_set and (f2, f1) not in gt_set]
    pairs = (gt_contrast + non_gt)[:6]
    if not pairs: return

    n_pairs, n_epochs = len(pairs), len(proxy_snaps)
    Z_all = {}
    global_vmin, global_vmax = np.inf, -np.inf

    for ei, snap in enumerate(proxy_snaps):
        m = MLP(num_features=len(FEATURE_COLS), hidden_units=hidden_dims, dropout=dropout).to(DEVICE)
        m.load_state_dict(snap.model_state); m.eval()
        w = ModelWrapper(m, scaler)
        for pi, (f1, f2, _) in enumerate(pairs):
            i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)
            pdp_res = partial_dependence(w, X_pdp, features=[(i1, i2)],
                                         grid_resolution=30, kind="average")
            g1, g2 = pdp_res["grid_values"]
            Z = pdp_res["average"][0]
            G1, G2 = np.meshgrid(g1, g2)
            Z_all[(pi, ei)] = (G1, G2, Z)
            global_vmin = min(global_vmin, Z.min())
            global_vmax = max(global_vmax, Z.max())

    levels = np.linspace(global_vmin, global_vmax, 20)
    fig, axes = plt.subplots(n_pairs, n_epochs,
                              figsize=(6*n_epochs, 5*n_pairs), squeeze=False)

    for pi, (f1, f2, cat) in enumerate(pairs):
        for ei, snap in enumerate(proxy_snaps):
            ax = axes[pi][ei]
            G1, G2, Z = Z_all[(pi, ei)]
            cs = ax.contourf(G1, G2, Z.T, levels=levels, cmap="viridis", alpha=0.8)
            ax.set_xlabel(f1, fontsize=9); ax.set_ylabel(f2, fontsize=9)
            if pi == 0: ax.set_title(f"Epoch {snap.epoch}", fontsize=11, fontweight="bold")
            if ei == 0:
                ax.annotate(f"{f1}×{f2}\n[{cat}]", xy=(-0.35, 0.5),
                            xycoords="axes fraction", fontsize=10,
                            ha="center", va="center", rotation=90)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(cs, cax=cbar_ax, label="Partial dependence")
    fig.suptitle(f"{func_label} – 2-D PDP time-lapse (shared colour scale)",
                 fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    p = _fname(output_dir, "12", "2d_pdp_timelapse", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


def plot_ice_faceted(snaps, hidden_dims, dropout, scaler, X_va_df,
                     func_label, output_dir, n_ice_samples=100):
    proxy_snaps = [s for s in snaps if s.model_state is not None]
    if not proxy_snaps:
        print("  [warn] no model states saved, skipping faceted ICE plots."); return

    pairs = GT_PAIRS[:3]
    if not pairs: return

    n_pairs, n_epochs = len(pairs), len(proxy_snaps)
    grid_pts = 50
    fig, axes = plt.subplots(n_pairs, n_epochs,
                              figsize=(6*n_epochs, 5*n_pairs), squeeze=False)

    X_ice     = X_va_df.sample(min(n_ice_samples, len(X_va_df)), random_state=42)
    X_ice_arr = X_ice.values.astype(np.float32)

    for ei, snap in enumerate(proxy_snaps):
        m = MLP(num_features=len(FEATURE_COLS), hidden_units=hidden_dims, dropout=dropout).to(DEVICE)
        m.load_state_dict(snap.model_state); m.eval()
        w = ModelWrapper(m, scaler)

        for pi, (f1, f2) in enumerate(pairs):
            ax = axes[pi][ei]
            i1, i2 = FEATURE_COLS.index(f1), FEATURE_COLS.index(f2)
            f1_vals, f2_vals = X_ice_arr[:, i1], X_ice_arr[:, i2]
            grid = np.linspace(np.percentile(f1_vals, 2), np.percentile(f1_vals, 98), grid_pts)
            f2_range = (f2_vals.max() - f2_vals.min()) or 1.0
            f2_norm  = (f2_vals - f2_vals.min()) / f2_range
            cmap     = plt.cm.coolwarm

            for obs_idx in range(len(X_ice_arr)):
                row_tiled = np.tile(X_ice_arr[obs_idx], (grid_pts, 1))
                row_tiled[:, i1] = grid
                preds = w.predict_proba(row_tiled)[:, 1]
                ax.plot(grid, preds, color=cmap(f2_norm[obs_idx]), alpha=0.3, linewidth=0.6)

            ax.set_xlabel(f1, fontsize=9); ax.set_ylabel("P(y=1)", fontsize=9)
            if pi == 0: ax.set_title(f"Epoch {snap.epoch}", fontsize=11, fontweight="bold")
            if ei == 0:
                ax.annotate(f"{f1}×{f2}", xy=(-0.35, 0.5),
                            xycoords="axes fraction", fontsize=10,
                            ha="center", va="center", rotation=90)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Conditioning feature value (normalised)")
    fig.suptitle(f"{func_label} – ICE plots (colour = conditioning feature)",
                 fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    p = _fname(output_dir, "13", "ice_faceted", func_label)
    plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {p.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN  –  loop over ALL 10 synth functions
# ══════════════════════════════════════════════════════════════════════════════

def run_one(func_idx: int, output_dir: Path) -> None:
    """Train and plot for one synth function under all EXPERIMENT_CONDITIONS."""
    func_label = f"F{func_idx + 1}"
    ph(f"===  {func_label}  (synth.functions[{func_idx}])  ===")

    set_seed(SEED)

    (X_tr, y_tr, y_tr_clean, X_va, y_va, X_te, y_te,
     X_va_raw, X_tr_raw, scaler, X_va_df, ground_truth) = load_synth_data(func_idx=func_idx)

    n_features = X_tr.shape[1]
    print(f"  Train={len(X_tr)}  Valid={len(X_va)}  Test={len(X_te)}")
    print(f"  Features={n_features}  Signal={SIGNAL_FEATURES}  Noise={NOISE_FEATURES}")
    print(f"  GT_PAIRS: {GT_PAIRS}")

    rng    = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_tr_raw), size=min(N_SHAP_BG, len(X_tr_raw)), replace=False)
    ex_idx = rng.choice(len(X_va_raw), size=min(N_SHAP_EXPLAIN, len(X_va_raw)), replace=False)
    X_bg     = X_tr_raw[bg_idx]
    X_ex_raw = X_va_raw[ex_idx]

    all_condition_snaps = []   # list of (cond_label, snaps)

    for cond_label, dropout, weight_decay in EXPERIMENT_CONDITIONS:
        ph(f"  {func_label} / {cond_label}  dropout={dropout}  wd={weight_decay}")
        model_name = f"{func_label}_{cond_label}"

        wrapper, snaps = train_model(
            n_features   = n_features,
            hidden_dims  = HIDDEN_DIMS,
            dropout      = dropout,
            weight_decay = weight_decay,
            X_tr=X_tr, y_tr=y_tr, y_tr_clean=y_tr_clean, X_va=X_va, y_va=y_va,
            X_bg=X_bg, X_ex=X_ex_raw, scaler=scaler,
            model_name=model_name,
        )

        last = snaps[-1]
        print(f"\n  {model_name}  train_auc={last.train_auc:.4f}  valid_auc={last.valid_auc:.4f}")
        all_condition_snaps.append((cond_label, snaps))

        label = f"{func_label}_{cond_label}"
        plot_training_curves(snaps, label, output_dir / func_label)
        plot_importance_evolution(snaps, label, output_dir / func_label)
        plot_noise_inflation(snaps, label, output_dir / func_label)
        plot_interaction_evolution(snaps, X_ex_raw, label, output_dir / func_label)
        plot_interaction_classification(wrapper, snaps, X_ex_raw, X_va_df,label, output_dir / func_label)
        # additional plots (previously unused)
        plot_h_statistic_evolution(snaps, HIDDEN_DIMS, dropout, scaler, X_va_df, label, output_dir / func_label)

    # Cross-condition noise inflation comparison (core Goal 2 evidence)
    if len(all_condition_snaps) > 1 and NOISE_FEATURES:
        fig, ax = plt.subplots(figsize=(10, 5))
        colours = ["steelblue", "tomato", "seagreen", "darkorange"]
        for (cond_label, snaps), colour in zip(all_condition_snaps, colours):
            shap_snaps = [s for s in snaps if s.shap_values is not None]
            if not shap_snaps:
                continue
            epochs = [s.epoch for s in shap_snaps]
            noise_idx = [FEATURE_COLS.index(c) for c in NOISE_FEATURES]
            noise_vals = [np.abs(s.shap_values[:, noise_idx]).mean() for s in shap_snaps]
            ax.plot(epochs, noise_vals, lw=2, color=colour, label=cond_label)
        ax.set_title(f"{func_label} – Noise SHAP inflation: regularized vs unregularized")
        ax.set_xlabel("Epoch"); ax.set_ylabel("mean |SHAP| noise features"); ax.legend()
        plt.tight_layout()
        p = _fname(output_dir, "00", "noise_inflation_comparison", func_label)
        plt.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
        print(f"  saved: {p.name}")

    ph(f"All outputs saved for {func_label}")
    for p in sorted((output_dir / func_label).rglob("*.png")):
        print(f"  {p.relative_to(output_dir / func_label)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for func_idx in ALL_FUNC_INDICES:
        run_one(func_idx, OUTPUT_DIR)
    ph("ALL FUNCTIONS COMPLETE")


if __name__ == "__main__":
    main()
