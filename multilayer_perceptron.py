import torch
import numpy as np
import torch.nn as nn
import copy


def get_weights(model):
    weights = []
    for name, param in model.named_parameters():
        if "interaction_mlp" in name and "weight" in name:
            weights.append(param.cpu().detach().numpy())
    return weights


class MLP(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_units,
        use_main_effect_nets=False,
        main_effect_net_units=[10, 10, 10],
    ):
        super(MLP, self).__init__()

        self.hidden_units = hidden_units
        self.use_main_effect_nets = use_main_effect_nets
        self.interaction_mlp = create_mlp([num_features] + hidden_units + [1])

        if main_effect_net_units == [1]:
            use_linear = True
        else:
            use_linear = False
        self.use_linear = use_linear

        if self.use_main_effect_nets:
            if use_linear:
                self.linear = nn.Linear(num_features, 1, bias=False)
            else:
                self.univariate_mlps = self.create_main_effect_nets(
                    num_features, main_effect_net_units, False, "uni"
                )

    def forward(self, x):
        y = self.interaction_mlp(x)

        if self.use_main_effect_nets:
            if self.use_linear:
                y += self.linear(x)
            else:
                y += self.forward_main_effect_nets(x, self.univariate_mlps)
        return y

    def create_main_effect_nets(self, num_features, hidden_units, out_bias, name):
        mlp_list = [
            create_mlp([1] + hidden_units + [1], out_bias=out_bias)
            for _ in range(num_features)
        ]
        for i in range(num_features):
            setattr(self, name + "_" + str(i), mlp_list[i])
        return mlp_list

    def forward_main_effect_nets(self, x, mlps):
        forwarded_mlps = []
        for i, mlp in enumerate(mlps):
            forwarded_mlps.append(mlp(x[:, [i]]))
        forwarded_mlp = sum(forwarded_mlps)
        return forwarded_mlp


def create_mlp(layer_sizes, out_bias=True):
    ls = list(layer_sizes)
    layers = nn.ModuleList()
    for i in range(1, len(ls) - 1):
        layers.append(nn.Linear(int(ls[i - 1]), int(ls[i])))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(int(ls[-2]), int(ls[-1]), bias=out_bias))
    return nn.Sequential(*layers)


def train(
    net,
    data_loaders,
    criterion=nn.MSELoss(reduction="mean"),
    nepochs=200,
    verbose=False,
    learning_rate=0.001,       # CHANGED: was 0.01 (SGD rate); Adam works well at 0.001–0.005
    device=torch.device("cpu"),
    save_snapshots=False,
    snapshot_epochs=None,
    snapshot_dir="./outputs/snapshots",
    sanity_check_every=0,
    l1_const=0.0,
    # ------------------------------------------------------------------ #
    # OVERFIT KNOBS — set defaults to the overfit-inducing configuration  #
    # ------------------------------------------------------------------ #
    use_adam=True,             # CHANGED: was False (SGD); Adam amplifies early spurious interactions
    weight_decay=0.0,          # CHANGED: ensure zero — any L2 dampens spurious interaction growth
    subsample_train=None,      # NEW: int or float (0,1) to shrink training set
    subsample_seed=0,          # NEW: seed for reproducible subsampling
):
    """
    Train the MLP model.

    Overfitting modifications vs. original
    ----------------------------------------
    1. Optimizer: SGD  →  Adam (adaptive LR amplifies front-loaded spurious interactions)
    2. weight_decay=0.0  (explicit zero; L2 regularisation dampens spurious interaction growth)
    3. subsample_train: optionally shrink the training set to reduce constraints on memorisation
       - int   → keep that many samples
       - float → keep that fraction of samples  e.g. 0.2 = 20 %

    Parameters
    ----------
    save_snapshots : bool
        If True, save model weights at specified epochs.
    snapshot_epochs : list of int, optional
        Epochs at which to save snapshots. If None and save_snapshots=True,
        defaults to [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300].
    snapshot_dir : str
        Directory to save snapshot files.
    sanity_check_every : int
        If > 0, run lightweight sanity checks every N epochs.
    l1_const : float
        L1 penalty applied to the interaction MLP weights only. This promotes
        sparse first-layer connectivity, which NID relies on to recover
        meaningful exact any-order interactions.
    use_adam : bool
        If True (default), use Adam optimizer. If False, use SGD (original behaviour).
    weight_decay : float
        L2 penalty. Keep at 0.0 to maximise overfitting.
    subsample_train : int | float | None
        Subsample the training DataLoader to a smaller dataset.
    subsample_seed : int
        Random seed for subsampling reproducibility.
    """
    import os

    # ------------------------------------------------------------------
    # MODIFICATION 3: optional training-set subsampling
    # ------------------------------------------------------------------
    if subsample_train is not None:
        train_dataset = data_loaders["train"].dataset
        n_total = len(train_dataset)

        if isinstance(subsample_train, float):
            n_keep = max(1, int(subsample_train * n_total))
        else:
            n_keep = int(subsample_train)

        rng = np.random.default_rng(subsample_seed)
        indices = rng.choice(n_total, size=n_keep, replace=False).tolist()
        subset = torch.utils.data.Subset(train_dataset, indices)

        # Preserve original DataLoader settings
        orig_loader = data_loaders["train"]
        data_loaders = dict(data_loaders)          # shallow copy — don't mutate caller's dict
        data_loaders["train"] = torch.utils.data.DataLoader(
            subset,
            batch_size=orig_loader.batch_size or 32,
            shuffle=True,
            num_workers=getattr(orig_loader, "num_workers", 0),
        )
        if verbose:
            print(f"[overfit] subsampled train set: {n_total} → {n_keep} samples")

    # ------------------------------------------------------------------
    # MODIFICATION 1 & 2: Adam instead of SGD, weight_decay=0.0
    # ------------------------------------------------------------------
    if use_adam:
        # Adam with weight_decay=0.0:
        #   - Adaptive per-parameter LR accelerates early memorisation
        #   - Zero weight decay removes the L2 brake on spurious interaction growth
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,   # 0.0 by default
        )
        if verbose:
            print(f"[overfit] using Adam  lr={learning_rate}  weight_decay={weight_decay}")
    else:
        # Original behaviour — pass use_adam=False to restore
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        if verbose:
            print(f"[overfit] using SGD   lr={learning_rate}  weight_decay={weight_decay}")

    history = {"epoch": [], "train_loss": [], "val_loss": []}
    snapshots = {}

    if save_snapshots:
        if snapshot_epochs is None:
            snapshot_epochs = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]
        snapshot_epochs = [e for e in snapshot_epochs if e <= nepochs]
        os.makedirs(snapshot_dir, exist_ok=True)

    def evaluate(net, data_loader, criterion, device):
        losses = []
        net.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = net(inputs)
                losses.append(criterion(preds, labels))
        return torch.stack(losses).mean()

    if verbose:
        print("starting to train")
        if save_snapshots:
            print(f"saving snapshots at epochs: {snapshot_epochs}")

    for epoch in range(nepochs):
        net.train()
        for inputs, labels in data_loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = net(inputs)
            loss = criterion(preds, labels)

            if l1_const > 0:
                l1_penalty = sum(
                    param.abs().sum()
                    for name, param in net.named_parameters()
                    if "interaction_mlp" in name and "weight" in name
                )
                loss = loss + l1_const * l1_penalty

            loss.backward()
            optimizer.step()

        current_epoch = epoch + 1

        tr = evaluate(net, data_loaders["train"], criterion, device).item()
        if "val" in data_loaders:
            va = evaluate(net, data_loaders["val"], criterion, device).item()
        else:
            va = float("nan")

        history["epoch"].append(current_epoch)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if sanity_check_every and current_epoch % sanity_check_every == 0:
            if not np.isfinite(tr) or not np.isfinite(va):
                raise ValueError(
                    f"Sanity check failed at epoch {current_epoch}: non-finite loss "
                    f"(train={tr}, val={va})."
                )

            with torch.no_grad():
                has_non_finite_param = any(
                    not torch.isfinite(p).all() for p in net.parameters()
                )
                param_sq_sum = sum((p.detach() ** 2).sum() for p in net.parameters())
                param_l2 = torch.sqrt(param_sq_sum).item()

            if has_non_finite_param:
                raise ValueError(
                    f"Sanity check failed at epoch {current_epoch}: non-finite model parameters."
                )

            msg = (
                f"[sanity] epoch {current_epoch}: "
                f"train={tr:.6f}, val={va:.6f}, param_l2={param_l2:.4f}"
            )
            if len(history["val_loss"]) >= 3:
                v0, v1, v2 = history["val_loss"][-3:]
                if np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2) and (v0 < v1 < v2):
                    msg += " | warning: val_loss increasing for 3 consecutive epochs"
            print(msg)

        if save_snapshots and current_epoch in snapshot_epochs:
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            snapshots[current_epoch] = state
            torch.save(state, os.path.join(snapshot_dir, f"model_epoch_{current_epoch}.pt"))

    if "test" in data_loaders:
        key = "test"
    elif "val" in data_loaders:
        key = "val"
    else:
        key = "train"
    test_loss = evaluate(net, data_loaders[key], criterion, device).item()

    if verbose:
        print("Finished Training. Test loss: ", test_loss)

    if save_snapshots:
        return net, test_loss, snapshots, history
    return net, test_loss, history


# =============================================================================
# USAGE EXAMPLE — maximum overfitting configuration
# =============================================================================
#
#   net = MLP(
#       num_features=10,
#       hidden_units=[256, 256, 256, 256],   # CHANGED: 4x wider & deeper than e.g. [64, 64]
#   )
#
#   net, test_loss, history = train(
#       net,
#       data_loaders,
#       nepochs=200,
#       learning_rate=0.005,     # higher than Adam default → faster early memorisation
#       use_adam=True,           # Adam instead of SGD
#       weight_decay=0.0,        # no L2 regularisation
#       subsample_train=0.2,     # use only 20 % of training data
#       verbose=True,
#   )
#
# To restore the original behaviour exactly:
#
#   net, test_loss, history = train(
#       net,
#       data_loaders,
#       learning_rate=0.01,
#       use_adam=False,
#       weight_decay=0.0,
#       subsample_train=None,
#   )
