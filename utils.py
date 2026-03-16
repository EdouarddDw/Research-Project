import torch
from torch.utils import data
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def force_float(X_numpy):
    return torch.from_numpy(X_numpy.astype(np.float32))


def convert_to_torch_loaders(Xd, Yd, batch_size):
    if type(Xd) != dict and type(Yd) != dict:
        Xd = {"train": Xd}
        Yd = {"train": Yd}

    data_loaders = {}
    for k in Xd:
        if k == "scaler":
            continue
        feats = force_float(Xd[k])
        targets = force_float(Yd[k])
        dataset = data.TensorDataset(feats, targets)
        data_loaders[k] = data.DataLoader(dataset, batch_size, shuffle=(k == "train"))

    return data_loaders


def preprocess_data(
    X,
    Y,
    valid_size=500,
    test_size=500,
    std_scale=False,
    get_torch_loaders=False,
    batch_size=100,
):

    n, p = X.shape
    ## Make dataset splits
    ntrain, nval, ntest = n - valid_size - test_size, valid_size, test_size

    Xd = {
        "train": X[:ntrain],
        "val": X[ntrain : ntrain + nval],
        "test": X[ntrain + nval : ntrain + nval + ntest],
    }
    Yd = {
        "train": np.expand_dims(Y[:ntrain], axis=1),
        "val": np.expand_dims(Y[ntrain : ntrain + nval], axis=1),
        "test": np.expand_dims(Y[ntrain + nval : ntrain + nval + ntest], axis=1),
    }

    for k in Xd:
        if len(Xd[k]) == 0:
            assert k != "train"
            del Xd[k]
            del Yd[k]

    if std_scale:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        scaler_x.fit(Xd["train"])
        scaler_y.fit(Yd["train"])

        for k in Xd:
            Xd[k] = scaler_x.transform(Xd[k])
            Yd[k] = scaler_y.transform(Yd[k])

        Xd["scaler"] = scaler_x
        Yd["scaler"] = scaler_y

    if get_torch_loaders:
        return convert_to_torch_loaders(Xd, Yd, batch_size)

    else:
        return Xd, Yd


def get_pairwise_auc(interactions, ground_truth):
    strengths = []
    gt_binary_list = []
    for inter, strength in interactions:
        inter_set = set(inter)  # assume 1-indexed
        strengths.append(strength)
        if any(inter_set <= gt for gt in ground_truth):
            gt_binary_list.append(1)
        else:
            gt_binary_list.append(0)

    auc = roc_auc_score(gt_binary_list, strengths)
    return auc


def get_anyorder_R_precision(interactions, ground_truth):

    R = len(ground_truth)
    recovered_gt = []
    counter = 0

    for inter, strength in interactions:
        if counter == R:
            break

        inter_set = set(inter)  # assume 1-indexed

        if any(inter_set < gt for gt in ground_truth):
            continue
        counter += 1
        if inter_set in ground_truth:
            recovered_gt.append(inter_set)

    R_precision = len(recovered_gt) / R

    return R_precision


def print_rankings(pairwise_interactions, anyorder_interactions, top_k=10, spacing=14):
    print(
        justify(["Pairwise interactions", "", "Arbitrary-order interactions"], spacing)
    )
    for i in range(top_k):
        p_inter, p_strength = pairwise_interactions[i]
        a_inter, a_strength = anyorder_interactions[i]
        print(
            justify(
                [
                    p_inter,
                    "{0:.4f}".format(p_strength),
                    "",
                    a_inter,
                    "{0:.4f}".format(a_strength),
                ],
                spacing,
            )
        )


def justify(row, spacing=14):
    return "".join(str(item).ljust(spacing) for item in row)


def load_snapshots(snapshot_dir: str, snapshot_epochs: list, device: str = "cpu") -> dict:
    """Load model snapshots from disk.
    
    Args:
        snapshot_dir: Directory containing snapshot .pt files.
        snapshot_epochs: List of epoch numbers to load.
        device: Device to map tensors to (default: 'cpu').
    
    Returns:
        Dictionary mapping epoch number to state_dict.
    """
    import os
    snapshots = {}
    for epoch in snapshot_epochs:
        snapshot_path = os.path.join(snapshot_dir, f"model_epoch_{epoch}.pt")
        if os.path.exists(snapshot_path):
            snapshots[epoch] = torch.load(snapshot_path, map_location=device, weights_only=True)
    return snapshots


def create_interaction_animation(
    pairwise_matrices: dict,
    anyorder_matrices: dict,
    available_epochs: list,
    labels: list,
    vmax_pw: float,
    vmax_ao: float,
    figsize: tuple = (14, 6),
):
    """Create side-by-side animated heatmaps for pairwise and any-order interactions.
    
    Args:
        pairwise_matrices: Dict mapping epoch -> pairwise interaction matrix.
        anyorder_matrices: Dict mapping epoch -> any-order interaction matrix.
        available_epochs: Sorted list of epoch numbers.
        labels: List of variable labels (e.g., ['x_1', 'x_2', ...]).
        vmax_pw: Max value for pairwise colorscale.
        vmax_ao: Max value for any-order colorscale.
        figsize: Figure size tuple.
    
    Returns:
        (fig, update_func) tuple for use with FuncAnimation.
    """
    import matplotlib.pyplot as plt
    
    num_features = len(labels)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Initial heatmaps
    im_pw = axes[0].imshow(pairwise_matrices[available_epochs[0]], cmap="viridis",
                           vmin=0, vmax=vmax_pw)
    im_ao = axes[1].imshow(anyorder_matrices[available_epochs[0]], cmap="magma",
                           vmin=0, vmax=vmax_ao)
    
    # Axis labels
    for ax in axes:
        ax.set_xticks(np.arange(num_features))
        ax.set_yticks(np.arange(num_features))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    axes[0].set_title("Pairwise Interactions")
    axes[1].set_title("Any-Order Interactions")
    
    fig.colorbar(im_pw, ax=axes[0], label="Strength", fraction=0.046, pad=0.04)
    fig.colorbar(im_ao, ax=axes[1], label="Strength", fraction=0.046, pad=0.04)
    
    title = fig.suptitle(f"Epoch {available_epochs[0]}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Text annotations
    annot_pw = [[axes[0].text(j, i, "", ha="center", va="center", fontsize=7)
                 for j in range(num_features)] for i in range(num_features)]
    annot_ao = [[axes[1].text(j, i, "", ha="center", va="center", fontsize=7)
                 for j in range(num_features)] for i in range(num_features)]
    
    def update(frame_idx):
        epoch = available_epochs[frame_idx]
        pw_mat = pairwise_matrices[epoch]
        ao_mat = anyorder_matrices[epoch]
        
        im_pw.set_data(pw_mat)
        im_ao.set_data(ao_mat)
        title.set_text(f"Epoch {epoch}")
        
        # Update annotations
        for i in range(num_features):
            for j in range(num_features):
                val_pw = pw_mat[i, j]
                val_ao = ao_mat[i, j]
                annot_pw[i][j].set_text(f"{val_pw:.1f}" if val_pw > 0.05 else "")
                annot_ao[i][j].set_text(f"{val_ao:.1f}" if val_ao > 0.05 else "")
                # Contrast color
                annot_pw[i][j].set_color("white" if val_pw > vmax_pw * 0.5 else "black")
                annot_ao[i][j].set_color("white" if val_ao > vmax_ao * 0.5 else "black")
        
        return [im_pw, im_ao, title] + [a for row in annot_pw for a in row] + [a for row in annot_ao for a in row]
    
    return fig, update


def detect_overfitting_epoch(
    epochs,
    train_losses,
    val_losses,
    patience=3,
    min_delta=0.0,
    smooth_window=3,
):
    """Detect overfitting onset from train/validation loss history."""
    epochs = np.asarray(epochs)
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)

    if len(epochs) == 0:
        raise ValueError("epochs must not be empty")

    if not (len(epochs) == len(train_losses) == len(val_losses)):
        raise ValueError("epochs, train_losses, and val_losses must have the same length")

    def moving_average(x, window):
        if window <= 1 or len(x) < window:
            return x.copy()
        kernel = np.ones(window) / window
        smoothed = np.convolve(x, kernel, mode="valid")
        pad_left = window // 2
        pad_right = len(x) - len(smoothed) - pad_left
        return np.pad(smoothed, (pad_left, pad_right), mode="edge")

    smooth_train = moving_average(train_losses, smooth_window)
    smooth_val = moving_average(val_losses, smooth_window)

    best_idx = int(np.argmin(smooth_val))
    best_val = smooth_val[best_idx]
    stale = 0
    onset_idx = best_idx

    for i in range(best_idx + 1, len(epochs)):
        val_improved = smooth_val[i] < (best_val - min_delta)
        train_not_worse = smooth_train[i] <= (smooth_train[i - 1] + min_delta)

        if val_improved:
            best_val = smooth_val[i]
            best_idx = i
            stale = 0
            continue

        stale = stale + 1 if train_not_worse else 0
        if stale >= patience:
            onset_idx = i - patience + 1
            break
    else:
        onset_idx = best_idx

    return int(epochs[onset_idx]), int(epochs[best_idx]), smooth_train, smooth_val


def compute_interaction_matrices_from_snapshots(
    snapshot_dir: str,
    snapshot_epochs: list,
    num_features: int,
    hidden_units: list,
    use_main_effect_nets: bool,
):
    """Load snapshots and compute pairwise/any-order interaction matrices."""
    from multilayer_perceptron import MLP, get_weights
    from neural_interaction_detection import get_interactions, interactions_to_matrix

    snapshots = load_snapshots(snapshot_dir, snapshot_epochs, device="cpu")
    if not snapshots:
        return {}, {}, []

    model = MLP(
        num_features,
        hidden_units,
        use_main_effect_nets=use_main_effect_nets,
    )
    model.eval()

    pairwise_matrices = {}
    anyorder_matrices = {}

    for epoch, state_dict in sorted(snapshots.items()):
        model.load_state_dict(state_dict)

        weights = get_weights(model)
        pairwise = get_interactions(weights, pairwise=True, one_indexed=True)
        anyorder = get_interactions(weights, pairwise=False, one_indexed=True)

        pairwise_matrices[epoch] = interactions_to_matrix(pairwise, num_features)
        anyorder_matrices[epoch] = interactions_to_matrix(anyorder, num_features)

    available_epochs = sorted(pairwise_matrices.keys())
    return pairwise_matrices, anyorder_matrices, available_epochs


def main():
    """Lightweight smoke checks for utility functions when run as a script."""
    set_seed(42)

    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    loaders = preprocess_data(
        X,
        y,
        valid_size=20,
        test_size=20,
        std_scale=False,
        get_torch_loaders=True,
        batch_size=16,
    )

    print("utils.py smoke check")
    print(f"splits: {list(loaders.keys())}")
    print(f"train batches: {len(loaders['train'])}")


if __name__ == "__main__":
    main()
