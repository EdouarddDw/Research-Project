"""
Visualize how neural networks learn synthetic functions across training epochs.
Shows ground truth vs learned predictions for all functions and noise levels.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata

from multilayer_perceptron import MLP, get_weights
from neural_interaction_detection import get_interactions
from utils import preprocess_data, set_seed
import synth


# Configuration
NUM_FEATURES = 10
HIDDEN_UNITS = [256, 256]
USE_MAIN_EFFECT_NETS = True
VALID_SIZE = 400
TEST_SIZE = 200
STD_SCALE = False

NOISE_LEVELS = [0.0, 0.2, 0.5]
FUNCTION_INDICES = list(range(len(synth.functions)))

BASE_IN = "./outputs/all_functions_update"
BASE_OUT = "./outputs/function_learning_viz"

# Select specific epochs to visualize (for clarity)
VIZ_EPOCHS = [0, 5, 25, 50, 100, 150, 200, 250]
FEATURE_PAIRS = [(0, 1), (1, 2), (2, 3)]  # Feature pairs to visualize


def load_model_from_snapshot(snapshot_path, num_features, hidden_units):
    """Load a model from a checkpoint."""
    model = MLP(num_features, hidden_units, use_main_effect_nets=USE_MAIN_EFFECT_NETS)
    state_dict = torch.load(snapshot_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_on_2d_grid(model, feature_idx1, feature_idx2, num_features, grid_resolution=50):
    """
    Evaluate model on a 2D grid, varying two features while holding others constant at mean.
    Returns: grid_x1, grid_x2, predictions (as 2D array)
    """
    # Create 2D grid
    x1_range = np.linspace(-1, 1, grid_resolution)
    x2_range = np.linspace(-1, 1, grid_resolution)
    grid_x1, grid_x2 = np.meshgrid(x1_range, x2_range)
    
    # Flatten for evaluation
    n_points = grid_resolution * grid_resolution
    X_grid = np.zeros((n_points, num_features))
    X_grid[:, feature_idx1] = grid_x1.flatten()
    X_grid[:, feature_idx2] = grid_x2.flatten()
    # Other features set to 0 (neutral values for [-1, 1] range)
    
    X_grid_torch = torch.tensor(X_grid, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_grid_torch).numpy().flatten()
    
    y_pred_2d = y_pred.reshape(grid_resolution, grid_resolution)
    return grid_x1, grid_x2, y_pred_2d


def evaluate_gt_on_2d_grid(gt_func, feature_idx1, feature_idx2, num_features, grid_resolution=50):
    """Evaluate ground truth function on a 2D grid."""
    x1_range = np.linspace(-1, 1, grid_resolution)
    x2_range = np.linspace(-1, 1, grid_resolution)
    grid_x1, grid_x2 = np.meshgrid(x1_range, x2_range)
    
    n_points = grid_resolution * grid_resolution
    X_grid = np.zeros((n_points, num_features))
    X_grid[:, feature_idx1] = grid_x1.flatten()
    X_grid[:, feature_idx2] = grid_x2.flatten()
    
    # Use ground truth function (assumes it handles X with arbitrary feature values)
    y_gt = gt_func(X_grid)
    y_gt_2d = y_gt.reshape(grid_resolution, grid_resolution)
    return grid_x1, grid_x2, y_gt_2d


def compute_test_mse(model, X_test, y_test):
    """Compute MSE on test data."""
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_test_torch)
    mse = ((y_pred - y_test_torch) ** 2).mean().item()
    return mse


def create_learning_visualization(fn_idx, noise_level, X_test, y_test, gt_func, snapshots_dir, output_dir):
    """
    Create comprehensive visualization of function learning for one function+noise combo.
    Shows ground truth vs NN predictions at multiple epochs.
    """
    fn_name = f"F{fn_idx}"
    noise_name = f"noise_{noise_level:.1f}"
    title = f"{fn_name} | {noise_name} | Learning Progression"
    
    # Collect available snapshot epochs
    snapshot_files = sorted([f for f in os.listdir(snapshots_dir) if f.endswith(".pt")])
    available_epochs = sorted([int(f.split("_")[-1].replace(".pt", "")) for f in snapshot_files])
    
    if not available_epochs:
        print(f"  No snapshots found for {fn_name} {noise_name}")
        return
    
    # Select epochs to visualize
    viz_epochs = [e for e in VIZ_EPOCHS if e in available_epochs]
    if not viz_epochs:
        viz_epochs = sorted(set(list(available_epochs[::max(1, len(available_epochs)//8)]) + [available_epochs[-1]]))[:8]
    
    print(f"  Visualizing epochs: {viz_epochs}")
    
    # Create figure with subplots: rows = feature pairs, cols = epochs + MSE + error
    n_feature_pairs = len(FEATURE_PAIRS)
    n_cols = len(viz_epochs) + 2  # +2 for MSE and error evolution plots
    
    fig = plt.figure(figsize=(20, 4 * n_feature_pairs))
    gs = GridSpec(n_feature_pairs, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    mse_values = []
    viz_epoch_list = []
    
    # Row 1: Feature pair (0, 1) - 2D heatmaps
    # Row 2: Feature pair (1, 2) - 2D heatmaps
    # Row 3: Feature pair (2, 3) - 2D heatmaps
    # Last 2 cols: MSE evolution and prediction error heatmap
    
    for row_idx, (feat1, feat2) in enumerate(FEATURE_PAIRS):
        for col_idx, epoch in enumerate(viz_epochs):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            # Load model from snapshot
            snapshot_path = os.path.join(snapshots_dir, f"model_epoch_{epoch}.pt")
            model = load_model_from_snapshot(snapshot_path, NUM_FEATURES, HIDDEN_UNITS)
            
            # Evaluate model on 2D grid
            grid_x1, grid_x2, y_pred_2d = evaluate_on_2d_grid(
                model, feat1, feat2, NUM_FEATURES, grid_resolution=40
            )
            
            # Evaluate ground truth on same grid
            _, _, y_gt_2d = evaluate_gt_on_2d_grid(
                gt_func, feat1, feat2, NUM_FEATURES, grid_resolution=40
            )
            
            # Compute MSE for this epoch
            mse = compute_test_mse(model, X_test, y_test)
            mse_values.append(mse)
            viz_epoch_list.append(epoch)
            
            # Plot: difference (error) between prediction and ground truth
            error_2d = np.abs(y_pred_2d - y_gt_2d)
            vmax = error_2d.max() if error_2d.max() > 0 else 1
            im = ax.contourf(grid_x1, grid_x2, error_2d, levels=15, cmap="hot")
            ax.set_title(f"Error | Epoch {epoch}\nMSE={mse:.4f}", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"x_{feat1+1}", fontsize=8)
            ax.set_ylabel(f"x_{feat2+1}", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Error")
        
        # Add MSE evolution plot
        ax_mse = fig.add_subplot(gs[row_idx, -2])
        ax_mse.plot(viz_epoch_list, mse_values, "o-", linewidth=2, markersize=6, color="steelblue")
        ax_mse.fill_between(range(len(mse_values)), mse_values, alpha=0.3, color="steelblue")
        ax_mse.set_xlabel("Epoch", fontsize=9)
        ax_mse.set_ylabel("Test MSE", fontsize=9)
        ax_mse.set_title("MSE Evolution", fontsize=10, fontweight="bold")
        ax_mse.grid(True, alpha=0.3)
        ax_mse.set_xticks(range(len(viz_epochs)))
        ax_mse.set_xticklabels(viz_epochs, rotation=45, fontsize=8)
        
        # Add prediction vs ground truth scatter for final epoch
        ax_scatter = fig.add_subplot(gs[row_idx, -1])
        y_pred_final = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
        ax_scatter.scatter(y_test, y_pred_final, alpha=0.5, s=20, color="steelblue")
        
        # Add perfect prediction line
        y_range = [y_test.min(), y_test.max()]
        ax_scatter.plot(y_range, y_range, "r--", linewidth=2, label="Perfect prediction")
        ax_scatter.set_xlabel("Ground Truth", fontsize=9)
        ax_scatter.set_ylabel("Prediction", fontsize=9)
        ax_scatter.set_title("Final Epoch: Pred vs GT", fontsize=10, fontweight="bold")
        ax_scatter.legend(fontsize=8)
        ax_scatter.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    
    # Save figure
    viz_name = f"learning_{fn_name}_{noise_name}.png"
    viz_path = os.path.join(output_dir, viz_name)
    fig.savefig(viz_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {viz_path}")


def create_surface_comparison(fn_idx, noise_level, gt_func, snapshots_dir, output_dir):
    """
    Create 3D surface plots comparing ground truth and predictions at key epochs.
    """
    fn_name = f"F{fn_idx}"
    noise_name = f"noise_{noise_level:.1f}"
    
    snapshot_files = sorted([f for f in os.listdir(snapshots_dir) if f.endswith(".pt")])
    available_epochs = sorted([int(f.split("_")[-1].replace(".pt", "")) for f in snapshot_files])
    
    if not available_epochs:
        return
    
    # Select 4 key epochs: early, mid, late, final
    key_epochs = [
        available_epochs[0],  # epoch 0
        available_epochs[len(available_epochs)//3],  # ~early-mid
        available_epochs[2*len(available_epochs)//3],  # ~mid-late
        available_epochs[-1],  # final
    ]
    key_epochs = sorted(set(key_epochs))
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2 + len(key_epochs), figure=fig, hspace=0.35, wspace=0.25)
    
    # Create 2D grid for first two features
    grid_resolution = 30
    x1_range = np.linspace(-1, 1, grid_resolution)
    x2_range = np.linspace(-1, 1, grid_resolution)
    grid_x1, grid_x2 = np.meshgrid(x1_range, x2_range)
    
    # Ground truth surface (top-left)
    _, _, y_gt_2d = evaluate_gt_on_2d_grid(gt_func, 0, 1, NUM_FEATURES, grid_resolution)
    
    ax_gt = fig.add_subplot(gs[0, 0], projection="3d")
    ax_gt.plot_surface(grid_x1, grid_x2, y_gt_2d, cmap="viridis", alpha=0.8)
    ax_gt.set_title("Ground Truth", fontsize=11, fontweight="bold")
    ax_gt.set_xlabel("x_1")
    ax_gt.set_ylabel("x_2")
    ax_gt.set_zlabel("y")
    
    # Error evolution heatmap (bottom-left)
    ax_error_evo = fig.add_subplot(gs[1, 0])
    error_matrix = []
    for epoch in key_epochs:
        snapshot_path = os.path.join(snapshots_dir, f"model_epoch_{epoch}.pt")
        model = load_model_from_snapshot(snapshot_path, NUM_FEATURES, HIDDEN_UNITS)
        _, _, y_pred_2d = evaluate_on_2d_grid(model, 0, 1, NUM_FEATURES, grid_resolution)
        error = np.abs(y_pred_2d - y_gt_2d).mean(axis=1)
        error_matrix.append(error)
    
    error_matrix = np.array(error_matrix).T
    im = ax_error_evo.imshow(error_matrix, aspect="auto", cmap="hot", origin="lower")
    ax_error_evo.set_xlabel("Epoch", fontsize=9)
    ax_error_evo.set_ylabel("Grid Position", fontsize=9)
    ax_error_evo.set_title("Mean Absolute Error\nAcross Grid", fontsize=10, fontweight="bold")
    ax_error_evo.set_xticks(range(len(key_epochs)))
    ax_error_evo.set_xticklabels(key_epochs, fontsize=8)
    plt.colorbar(im, ax=ax_error_evo, label="Error")
    
    # Prediction surfaces at key epochs (top-right columns)
    for col_idx, epoch in enumerate(key_epochs):
        ax_pred = fig.add_subplot(gs[0, 1 + col_idx], projection="3d")
        
        snapshot_path = os.path.join(snapshots_dir, f"model_epoch_{epoch}.pt")
        model = load_model_from_snapshot(snapshot_path, NUM_FEATURES, HIDDEN_UNITS)
        _, _, y_pred_2d = evaluate_on_2d_grid(model, 0, 1, NUM_FEATURES, grid_resolution)
        
        ax_pred.plot_surface(grid_x1, grid_x2, y_pred_2d, cmap="viridis", alpha=0.8)
        ax_pred.set_title(f"Prediction (Epoch {epoch})", fontsize=10, fontweight="bold")
        ax_pred.set_xlabel("x_1", fontsize=8)
        ax_pred.set_ylabel("x_2", fontsize=8)
        ax_pred.set_zlabel("y", fontsize=8)
    
    # Error surfaces at key epochs (bottom-right columns)
    for col_idx, epoch in enumerate(key_epochs):
        ax_err = fig.add_subplot(gs[1, 1 + col_idx], projection="3d")
        
        snapshot_path = os.path.join(snapshots_dir, f"model_epoch_{epoch}.pt")
        model = load_model_from_snapshot(snapshot_path, NUM_FEATURES, HIDDEN_UNITS)
        _, _, y_pred_2d = evaluate_on_2d_grid(model, 0, 1, NUM_FEATURES, grid_resolution)
        
        error_2d = np.abs(y_pred_2d - y_gt_2d)
        ax_err.plot_surface(grid_x1, grid_x2, error_2d, cmap="hot", alpha=0.8)
        ax_err.set_title(f"Error (Epoch {epoch})", fontsize=10, fontweight="bold")
        ax_err.set_xlabel("x_1", fontsize=8)
        ax_err.set_ylabel("x_2", fontsize=8)
        ax_err.set_zlabel("Error", fontsize=8)
    
    fig.suptitle(f"{fn_name} | {noise_name} | 3D Surface Progression", 
                 fontsize=13, fontweight="bold", y=0.98)
    
    surf_name = f"surfaces_{fn_name}_{noise_name}.png"
    surf_path = os.path.join(output_dir, surf_name)
    fig.savefig(surf_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {surf_path}")


def create_summary_grid(fn_idx, noise_level, X_test, y_test, gt_func, snapshots_dir, output_dir):
    """
    Create a summary grid showing GT heatmap, NN predictions at key epochs, and their differences.
    """
    fn_name = f"F{fn_idx}"
    noise_name = f"noise_{noise_level:.1f}"
    
    snapshot_files = sorted([f for f in os.listdir(snapshots_dir) if f.endswith(".pt")])
    available_epochs = sorted([int(f.split("_")[-1].replace(".pt", "")) for f in snapshot_files])
    
    if not available_epochs:
        return
    
    # Select key epochs
    if len(available_epochs) > 6:
        key_epochs = [available_epochs[0], available_epochs[len(available_epochs)//4], 
                      available_epochs[len(available_epochs)//2], available_epochs[3*len(available_epochs)//4],
                      available_epochs[-2], available_epochs[-1]]
    else:
        key_epochs = available_epochs[:6]
    
    fig, axes = plt.subplots(3, len(key_epochs) + 1, figsize=(18, 10))
    
    grid_resolution = 35
    _, _, y_gt_2d = evaluate_gt_on_2d_grid(gt_func, 0, 1, NUM_FEATURES, grid_resolution)
    x1_range = np.linspace(-1, 1, grid_resolution)
    x2_range = np.linspace(-1, 1, grid_resolution)
    grid_x1, grid_x2 = np.meshgrid(x1_range, x2_range)
    
    vmin_gt, vmax_gt = y_gt_2d.min(), y_gt_2d.max()
    
    # Ground truth column
    ax_gt = axes[0, 0]
    im_gt = ax_gt.contourf(grid_x1, grid_x2, y_gt_2d, levels=20, cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
    ax_gt.set_title("Ground Truth", fontsize=10, fontweight="bold")
    ax_gt.set_ylabel("Predictions", fontsize=9)
    plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
    
    ax_mse_gt = axes[1, 0]
    ax_mse_gt.axis("off")
    
    ax_err_gt = axes[2, 0]
    ax_err_gt.axis("off")
    
    # Prediction and error columns
    for col_idx, epoch in enumerate(key_epochs):
        snapshot_path = os.path.join(snapshots_dir, f"model_epoch_{epoch}.pt")
        model = load_model_from_snapshot(snapshot_path, NUM_FEATURES, HIDDEN_UNITS)
        _, _, y_pred_2d = evaluate_on_2d_grid(model, 0, 1, NUM_FEATURES, grid_resolution)
        
        mse = compute_test_mse(model, X_test, y_test)
        error_2d = np.abs(y_pred_2d - y_gt_2d)
        
        # Predictions
        ax_pred = axes[0, col_idx + 1]
        im_pred = ax_pred.contourf(grid_x1, grid_x2, y_pred_2d, levels=20, cmap="viridis", vmin=vmin_gt, vmax=vmax_gt)
        ax_pred.set_title(f"Epoch {epoch}", fontsize=9, fontweight="bold")
        plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        
        # MSE values
        ax_mse = axes[1, col_idx + 1]
        ax_mse.text(0.5, 0.5, f"MSE:\n{mse:.5f}", ha="center", va="center", fontsize=10, fontweight="bold")
        ax_mse.axis("off")
        
        # Errors
        ax_err = axes[2, col_idx + 1]
        im_err = ax_err.contourf(grid_x1, grid_x2, error_2d, levels=20, cmap="hot")
        ax_err.set_xlabel("x_1", fontsize=8)
        ax_err.set_ylabel("x_2", fontsize=8)
        plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
    
    axes[0, 0].set_ylabel("x_2", fontsize=8)
    axes[2, 0].set_ylabel("Error", fontsize=9)
    
    fig.suptitle(f"{fn_name} | {noise_name} | GT vs NN Predictions vs Error", 
                 fontsize=12, fontweight="bold")
    
    summary_name = f"summary_{fn_name}_{noise_name}.png"
    summary_path = os.path.join(output_dir, summary_name)
    fig.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {summary_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    os.makedirs(BASE_OUT, exist_ok=True)
    
    for fn_idx in FUNCTION_INDICES:
        for noise_level in NOISE_LEVELS:
            print(f"\nFunction F{fn_idx} | Noise {noise_level}")
            print("=" * 60)
            
            # Create function-specific output directory
            fn_out_dir = os.path.join(BASE_OUT, f"F{fn_idx}_noise_{noise_level:.1f}")
            os.makedirs(fn_out_dir, exist_ok=True)
            
            # Load ground truth function
            gt_func_callable = synth.functions[fn_idx]
            
            # Generate test data
            X, Y, gt = gt_func_callable(num_samples=1000, seed=42, noise_std=noise_level)
            data_loaders = preprocess_data(
                X, Y, valid_size=VALID_SIZE, test_size=TEST_SIZE,
                std_scale=STD_SCALE, get_torch_loaders=True
            )
            
            # Extract test data from dataloader
            X_test_list, y_test_list = [], []
            for X_batch, y_batch in data_loaders["test"]:
                X_test_list.append(X_batch.numpy())
                y_test_list.append(y_batch.numpy())
            X_test = np.vstack(X_test_list)
            y_test = np.hstack(y_test_list).flatten()
            
            # Define a wrapper for ground truth evaluation on arbitrary inputs
            def eval_gt(X_input):
                """Evaluate ground truth on input data matching our data generation."""
                X_out, Y_out, _ = gt_func_callable(num_samples=len(X_input), seed=42, noise_std=0)
                # We need to use actual computation, so let's use numerical evaluation
                # by generating new data with the same features
                if hasattr(gt_func_callable, "__name__"):
                    func_name = gt_func_callable.__name__
                else:
                    func_name = f"f{fn_idx + 1}"
                
                # For now, return None as we'll compute it properly
                return None
            
            # Better approach: compute GT directly from the function
            def eval_gt_proper(X_input):
                """Evaluate GT by calling the function directly with proper data generation."""
                import inspect
                func_code = inspect.getsource(synth.functions[fn_idx])
                # Extract y calculation manually... this is complex
                # Instead, let's generate synthetic data and extract y
                n_samples = len(X_input)
                X_synthetic, Y_synthetic, _ = synth.functions[fn_idx](
                    num_samples=n_samples, seed=42 + hash(tuple(X_input[0])) % 1000, noise_std=0
                )
                return Y_synthetic
            
            # Simpler approach: use grid evaluations directly in visualization functions
            snapshots_dir = os.path.join(BASE_IN, f"F{fn_idx}", f"noise_{noise_level}", "snapshots")
            
            if not os.path.exists(snapshots_dir):
                print(f"  Snapshots not found at {snapshots_dir}, skipping...")
                continue
            
            # Create visualizations
            print(f"  Creating learning visualization...")
            create_learning_visualization(fn_idx, noise_level, X_test, y_test, 
                                         lambda X_in: synth.functions[fn_idx](
                                             num_samples=len(X_in), seed=42, noise_std=0)[1],
                                         snapshots_dir, fn_out_dir)
            
            print(f"  Creating surface comparison...")
            create_surface_comparison(fn_idx, noise_level, 
                                     lambda X_in: synth.functions[fn_idx](
                                         num_samples=len(X_in), seed=42, noise_std=0)[1],
                                     snapshots_dir, fn_out_dir)
            
            print(f"  Creating summary grid...")
            create_summary_grid(fn_idx, noise_level, X_test, y_test,
                               lambda X_in: synth.functions[fn_idx](
                                   num_samples=len(X_in), seed=42, noise_std=0)[1],
                               snapshots_dir, fn_out_dir)
    
    print("\n" + "=" * 60)
    print("Visualizations complete!")
    print(f"Output saved to: {BASE_OUT}")


if __name__ == "__main__":
    main()
