import torch
from torch.utils import data
import numpy as np
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
