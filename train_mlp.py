import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, random_split

from multilayer_perceptron import MLP, train
import synth

SEED = 42
N_SAMPLES = 30000 # adjust to 30000 for final runs
NOISE_STD =0.1
np.random.seed(SEED)
torch.manual_seed(SEED)
snapshot_root = "./outputs/snapshots/f3"
os.makedirs(snapshot_root, exist_ok=True)

SNAPSHOT_EPOCHS = sorted(
    [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300]
)

def build_dataloaders(X, Y, batch_size=128):

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, Y)

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    data_loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size),
        "test": DataLoader(test_ds, batch_size=batch_size),
    }

    return data_loaders


def run_training(run_name, l2_const=0):

    print(f"\nTraining run: {run_name}")

    X, Y, ground_truth = synth.functions[3](num_samples=N_SAMPLES, seed=SEED, noise_std=NOISE_STD)

    data_loaders = build_dataloaders(X, Y)

    net = MLP(
        num_features=X.shape[1],
        hidden_units=[140, 100, 60, 20],
        use_main_effect_nets=False
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = net.to(device)

    snapshot_dir = os.path.join(snapshot_root, run_name)

    net, test_loss, snapshots = train(
        net,
        data_loaders,
        nepochs=300,
        learning_rate=0.01,
        verbose=True,
        device=device,
        l2_const=l2_const,
        snapshot_dir=snapshot_dir,
        snapshot_epochs=SNAPSHOT_EPOCHS
    )

    print("Final test loss:", test_loss)

def main():
    run_training("l1", l2_const=0)

if __name__ == "__main__":
    main()