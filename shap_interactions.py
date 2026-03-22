import torch
import numpy as np
import os
import shap

from multilayer_perceptron import MLP

# match your training config
EXPERIMENT = "noise1_lr0.01_epochs 300_samples3k"

snapshot_root = "./outputs/snapshots/"
data_root = "./outputs/data/"


#############################################
# Interaction metric (stable)
#############################################

def pairwise_interaction_matrix(shap_values):

    n_features = shap_values.shape[1]
    matrix = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i + 1, n_features):

            interaction = np.mean(np.abs(shap_values[:, i] * shap_values[:, j]))

            matrix[i, j] = interaction
            matrix[j, i] = interaction

    return matrix


#############################################
# SHAP computation (KernelExplainer ONLY)
#############################################

def get_shap_interactions(model, X):

    model.eval()

    def predict_fn(x):
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x)
        return preds.detach().numpy().flatten()   # CRITICAL FIX

    explainer = shap.KernelExplainer(predict_fn, X[:100])

    shap_values = explainer.shap_values(X[:150])

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values)

    return pairwise_interaction_matrix(shap_values)


#############################################
# Run SHAP for one function
#############################################

def run_shap_for_function(func_id):

    print(f"\nRunning SHAP for f{func_id+1}")

    snapshot_dir = os.path.join(
        snapshot_root,
        f"f{func_id+1}",
        EXPERIMENT
    )

    data_dir = os.path.join(
        data_root,
        EXPERIMENT,
        f"f{func_id+1}"
    )

    X_val = np.load(os.path.join(data_dir, "X_val.npy"))

    for file in sorted(os.listdir(snapshot_dir)):

        if not file.endswith(".pt"):
            continue

        epoch = int(file.split("_")[-1].replace(".pt", ""))

        print("Epoch:", epoch)

        model = MLP(
            num_features=X_val.shape[1],
            hidden_units=[256,256],
            use_main_effect_nets=False
        )

        model.load_state_dict(
            torch.load(os.path.join(snapshot_dir, file), map_location="mps")
        )

        interaction_matrix = get_shap_interactions(model, X_val)

        # save matrix
        np.save(
            os.path.join(snapshot_dir, f"shap_interactions_epoch_{epoch}.npy"),
            interaction_matrix
        )
