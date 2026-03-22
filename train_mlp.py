import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from multilayer_perceptron import MLP, train
from synth import NUM_SAMPLES, SEED, NOISE_STD
import synth


np.random.seed(SEED)
torch.manual_seed(SEED)
snapshot_root = "./outputs/snapshots/"
os.makedirs(snapshot_root, exist_ok=True)

SNAPSHOT_EPOCHS = sorted(
    [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250]
)

def build_dataloaders(X, Y, batch_size=32):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.30, random_state=SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    data_loaders = {
        "train": DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True
        ),
        "val": DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size
        ),
        "test": DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=batch_size
        ),
    }
    return data_loaders, (X_train, X_val, X_test, y_train, y_val, y_test)

def run_training(func_id):
    print(f"\nTraining function f{func_id + 1}")

    func = synth.functions[func_id]

    X, Y, ground_truth = func(
        num_samples=NUM_SAMPLES,
        seed=SEED,
        noise_std=NOISE_STD
    )

    data_loaders, splits = build_dataloaders(X, Y)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    data_dir = os.path.join(
        "./outputs/data/noise1_lr0.01_epochs 300_samples3k",
        f"f{func_id + 1}"
    )
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)

    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    net = MLP(
        num_features=X.shape[1],
        hidden_units=[256, 256, 128, 64],  # 🔴 increased capacity
        use_main_effect_nets=False
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net = net.to(device)

    snapshot_dir = os.path.join(
        snapshot_root,
        f"f{func_id + 1}",
        "noise1_lr0.01_epochs 300_samples3k"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    net, test_loss, snapshots = train(
        net,
        data_loaders,
        nepochs=250,
        verbose=True,
        learning_rate=0.01,
        snapshot_dir=snapshot_dir,
        snapshot_epochs=SNAPSHOT_EPOCHS
    )

    print("Final test loss:", test_loss)

def main():
    for func_id in range(len(synth.functions)):
        run_training(func_id)

if __name__ == "__main__":
    main()