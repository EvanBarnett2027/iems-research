# main.py
import numpy as np
import tensorflow as tf
import gpflow
import matplotlib.pyplot as plt
import importlib # To dynamically import simulator modules

# --- User Configuration ---
SIMULATOR_NAME = 'Multivariate'
# SIMULATOR_NAME = 'OTLcircuit'
# SIMULATOR_NAME = 'Borehole' 
# SIMULATOR_NAME = 'Wingweight'

MODEL_TYPE = 'OAK'
# MODEL_TYPE = 'StandardGP'

# Simulation Design Params
m_train = 100
m_test = 50
n_plot_points_smooth = 101 # For the smooth plot vs input

# PCGP Specific Config
num_pcs_to_use = None #Or set to None to use variance threshold

OAK_CONFIG = {
    'max_interaction_depth': 4, # Use 2 for MultiSine example (total_input_dim=4)
    'num_inducing': 100,        # Adjusted based on m_train
    'use_sparsity_prior': False,
    'use_normalising_flow': False, # Using standardization
    'share_var_across_orders': True,
    'sparse': False # m_train is small
}

# Plotting Config
plot_validation_metrics = True # Print RMSE/MAE etc.
plot_actual_vs_pred = True   # Generate the Actual vs. Predicted plot (type depends on n_eta)
plot_pred_vs_input = True    # Generate the Prediction vs. Input plot (type depends on n_eta)
plot_reconstructed_output = True # Add a plot for reconstructed output if multivariate
plot_pca_variance = True # Plot PCA variance if applicable

# Plotting Config - Specific Plot Options
var_idx_to_plot_smooth = 0 # Index (0 to total_input_dim-1) for the smooth plot
fixed_val_smooth = 0.5     # Standardized value for other inputs in smooth plot
test_point_to_plot_recon = 0 # Index of test point for reconstructed plot (only if n_eta > 1)
# --- End User Configuration ---

# --- Conditional Import for OAK ---
if MODEL_TYPE == 'OAK':
    try:
        from oak.model_utils import oak_model
        print("OAK module loaded successfully.")
    except ImportError as e:
        print(f"Error importing OAK module: {e}")
        exit()
# --- End Conditional Import ---

# --- Setup based on SIMULATOR_NAME ---
print(f"--- Loading Simulator: {SIMULATOR_NAME} ---")

simulator_module_name = f"{SIMULATOR_NAME.lower()}_simulator" 
try:
    simulator_module = importlib.import_module(simulator_module_name)
    run_simulator = getattr(simulator_module, f"run_simulator_{SIMULATOR_NAME.lower()}")
    simulator_metadata = getattr(simulator_module, "_dict")
except (ImportError, AttributeError) as e:
    print(f"Error loading simulator '{SIMULATOR_NAME}': {e}")
    print(f"Ensure '{simulator_module_name}.py' exists and contains 'run_simulator_{SIMULATOR_NAME.lower()}' and '_dict'.")
    exit()

# Extract dimensions from metadata
xdim = simulator_metadata['xdim']
thetadim = simulator_metadata['thetadim']
total_input_dim = xdim + thetadim
n_eta = simulator_metadata.get('n_eta_out', 1)
simulator_func_name = simulator_metadata.get('function', SIMULATOR_NAME) # Use name from dict if available
# --- End Simulator Setup ---


# --- Emulator Utils Import ---
try:
    # Use the new generalized function names
    from pcgp_utils import train_emulator, predict_emulator
except ImportError:
    print("Error: Could not import from pcgp_utils.py. Make sure the file exists.")
    exit()
# --- End Emulator Utils Import ---


# --- GPFlow/Seed Config ---
gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_summary_fmt("notebook")
np.random.seed(123)
tf.random.set_seed(123)
print(f"Simulator '{simulator_func_name}' loaded.")
print(f"Input dimensions: x={xdim}, theta={thetadim}, total={total_input_dim}")
print(f"Output dimension: n_eta={n_eta}")
print("--- Setup Complete ---")


# --- Main Workflow ---
if __name__ == "__main__":

    # 1. Generate Standardized Training Design [0, 1]
    sim_inputs_train_std = np.random.rand(m_train, total_input_dim)

     # 2. Run Selected Simulator for Training Data
    print(f"\n--- Running Simulator ({simulator_func_name}) for Training Data ---")
    sim_outputs_train = run_simulator(sim_inputs_train_std)
    # Ensure output is 2D (m, n_eta)
    if sim_outputs_train.ndim == 1:
        sim_outputs_train = sim_outputs_train.reshape(-1, 1)
    print(f"Simulator training output shape: {sim_outputs_train.shape}")
    # Update n_eta based on actual output if necessary (safety check)
    
    if n_eta != sim_outputs_train.shape[1]:
        print(f"Warning: n_eta in metadata ({n_eta}) differs from actual output dim ({sim_outputs_train.shape[1]}). Using actual.")
        n_eta = sim_outputs_train.shape[1]

    # 3. Train the PCGP Emulator
    # Pass standardized inputs to training function
    emulator_models, pca_obj, input_scaler_obj, output_scaler_obj = train_emulator(
        sim_inputs_std=sim_inputs_train_std,
        sim_outputs=sim_outputs_train,
        model_type=MODEL_TYPE,
        num_pcs=num_pcs_to_use if n_eta > 1 else None, # Only pass num_pcs if relevant
        oak_config=OAK_CONFIG if MODEL_TYPE == 'OAK' else None # Only pass oak_config if relevant
    )

    if emulator_models is None:
        print("Emulator training failed. Exiting.")
        exit()

    # --- Validation & Plotting ---
    if m_test > 0:
        print("\n--- Validation Phase ---")
        # 4. Generate Standardized Test Design [0, 1]
        test_inputs_std = np.random.rand(m_test, total_input_dim)

        # 5. Make Predictions using the Emulator
        print("Making predictions on test data...")
        # Pass all necessary objects returned from train_emulator
        pred_mean, pred_var = predict_emulator(
            new_inputs_std=test_inputs_std,
            emulator_models=emulator_models, # List of GPR or OAK instances
            model_type=MODEL_TYPE,
            pca_obj=pca_obj,
            input_scaler=input_scaler_obj,
            output_scaler=output_scaler_obj
        )
        
        # Ensure predictions are valid before proceeding
        if pred_mean is None or pred_var is None:
             print("Prediction failed. Skipping further validation and plotting.")
        else:
            if pred_mean.ndim == 1: pred_mean = pred_mean.reshape(-1, 1)
            if pred_var.ndim == 1: pred_var = pred_var.reshape(-1, 1)

        # 6. Run Actual Simulator for Validation
        print("Running simulator for test points for validation...")
        actual_outputs_test = run_simulator(test_inputs_std)
        if actual_outputs_test.ndim == 1:
             actual_outputs_test = actual_outputs_test.reshape(-1, 1)

        # Use the actual n_eta from prediction output shape
        current_n_eta = pred_mean.shape[1]

        # 7. Calculate & Print Validation Stats
        if plot_validation_metrics:
                if current_n_eta == 1: # Scalar Metrics
                    rmse = np.sqrt(np.mean((actual_outputs_test[:,0] - pred_mean[:,0])**2))
                    mean_abs_err = np.mean(np.abs(actual_outputs_test[:,0] - pred_mean[:,0]))
                    avg_pred_std_dev = np.mean(np.sqrt(pred_var[:,0]))
                    print(f"\nValidation Results ({m_test} points, Output Dim 0):")
                    print(f"  RMSE: {rmse:.4g}")
                    print(f"  MAE:  {mean_abs_err:.4g}")
                    print(f"  Avg. Pred. Std Dev: {avg_pred_std_dev:.4g}")
                else: # Multivariate metrics
                    rmse_per_point = np.sqrt(np.mean((actual_outputs_test - pred_mean)**2, axis=1))
                    avg_rmse = np.mean(rmse_per_point)
                    avg_pred_std_dev = np.mean(np.sqrt(pred_var))
                    print(f"\nValidation Results ({m_test} points, {current_n_eta} Dims):")
                    print(f"  Avg. RMSE across output dims: {avg_rmse:.4g}")
                    print(f"  Avg. Pred. Std Dev across output dims: {avg_pred_std_dev:.4g}")
        # 8. Plotting
        print("\n--- Generating Plots ---")
        plot_title_suffix = f"({simulator_func_name} using {MODEL_TYPE})"

         # === PLOT TYPE 1: Actual vs Predicted ===
        if plot_actual_vs_pred:
            # (Same plotting logic as before, dynamically adapts based on current_n_eta)
            plt.figure(figsize=(9, 8))
            if current_n_eta == 1:
                print("Generating Actual vs Predicted plot (Scalar version)...")
                std_dev = np.sqrt(pred_var[:,0])
                lower_95 = pred_mean[:,0] - 1.96 * std_dev
                upper_95 = pred_mean[:,0] + 1.96 * std_dev
                actual_flat = actual_outputs_test[:,0]
                mean_flat = pred_mean[:,0]
                sort_indices = np.argsort(actual_flat)
                sorted_actual = actual_flat[sort_indices]
                sorted_mean = mean_flat[sort_indices]
                sorted_lower = lower_95[sort_indices]
                sorted_upper = upper_95[sort_indices]
                min_val = min(actual_flat.min(), mean_flat.min())
                max_val = max(actual_flat.max(), mean_flat.max())
                range_padding = (max_val - min_val) * 0.1
                plot_min = min_val - range_padding if np.isfinite(min_val) else -10
                plot_max = max_val + range_padding if np.isfinite(max_val) else 10
                plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=1.5, alpha=0.7, label='Ideal (y=x)')
                plt.fill_between(sorted_actual, sorted_lower, sorted_upper, color='lightskyblue', alpha=0.5, label='Emulator 95% CI')
                plt.plot(sorted_actual, sorted_mean, '-', color='dodgerblue', lw=2, label='Emulator Mean Prediction')
                plt.title(f"Emulator Prediction vs Actual {plot_title_suffix}", fontsize=14)
                plt.xlabel(f"Actual Output", fontsize=12)
                plt.ylabel(f"Emulator Mean Prediction", fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.xlim(plot_min, plot_max)
                plt.ylim(plot_min, plot_max)
                plt.tight_layout()
                plt.show()
            else:
                print("Generating Actual vs Predicted plot (Multivariate Norm version)...")
                norm_actual = np.linalg.norm(actual_outputs_test, axis=1)
                norm_pred = np.linalg.norm(pred_mean, axis=1)
                plt.scatter(norm_actual, norm_pred, alpha=0.7, c='dodgerblue', label='Test Points (Norms)')
                min_val = 0
                max_val = max(norm_actual.max(), norm_pred.max()) * 1.05
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal (y=x)')
                plt.title(f"Norm of Emulator Prediction vs Actual {plot_title_suffix}", fontsize=14)
                plt.xlabel(f"Norm of Actual Output", fontsize=12)
                plt.ylabel(f"Norm of Emulator Mean Prediction", fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.tight_layout()
                plt.show()


        # === PLOT TYPE 2: Prediction vs Single Input ===
        if plot_pred_vs_input:
                # (Plotting logic remains the same, plots first output dim if multivariate)
            if var_idx_to_plot_smooth >= total_input_dim: print(f"\nError: var_idx_to_plot_smooth invalid.")
            else:
                print(f"\nGenerating plot of Emulator Prediction vs. input index {var_idx_to_plot_smooth}...")
                plot_inputs_std = np.full((n_plot_points_smooth, total_input_dim), fixed_val_smooth)
                varying_input_std = np.linspace(0, 1, n_plot_points_smooth)
                plot_inputs_std[:, var_idx_to_plot_smooth] = varying_input_std
                plot_pred_mean, plot_pred_var = predict_emulator(plot_inputs_std, emulator_models, MODEL_TYPE, pca_obj, input_scaler_obj, output_scaler_obj)
                if plot_pred_mean is not None:
                        # (Rest of plotting code is identical to previous version)
                    plot_mean_to_show = plot_pred_mean[:, 0]
                    plot_var_to_show = plot_pred_var[:, 0]
                    y_label_suffix = " (Dim 0)" if current_n_eta > 1 else ""
                    plot_std_dev = np.sqrt(np.maximum(plot_var_to_show, 1e-18))
                    plot_lower_95 = plot_mean_to_show - 1.96 * plot_std_dev
                    plot_upper_95 = plot_mean_to_show + 1.96 * plot_std_dev
                    plt.figure(figsize=(8, 5))
                    plt.plot(varying_input_std, plot_mean_to_show, 'b', lw=2, label=f'Emulator Mean{y_label_suffix}')
                    plt.fill_between(varying_input_std, plot_lower_95, plot_upper_95, color='lightblue', alpha=0.5, label=f'Emulator 95% CI{y_label_suffix}')
                    if var_idx_to_plot_smooth < xdim: input_var_name = f'x_std[{var_idx_to_plot_smooth+1}]'
                    else: input_var_name = f'theta_std[{var_idx_to_plot_smooth - xdim + 1}]'
                    plt.title(f"Emulator Prediction {plot_title_suffix} vs. Input '{input_var_name}'", fontsize=14)
                    plt.xlabel(f"Standardized Input '{input_var_name}' (Others Fixed at {fixed_val_smooth})", fontsize=12)
                    plt.ylabel(f"Emulator Predicted Output{y_label_suffix}", fontsize=12)
                    plt.legend(fontsize=10)
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.tight_layout()
                    plt.show()


        # === PLOT TYPE 3: Reconstructed Output (Multivariate Only) ===
        if current_n_eta > 1:
                # (Plotting logic remains the same as before)
            if test_point_to_plot_recon >= m_test: print(f"Warning: test_point_to_plot_recon invalid.")
            else:
                print(f"\nGenerating plot of reconstructed output for test point {test_point_to_plot_recon}...")
                actual_single = actual_outputs_test[test_point_to_plot_recon, :]
                pred_mean_single = pred_mean[test_point_to_plot_recon, :]
                pred_var_single = pred_var[test_point_to_plot_recon, :]
                pred_std_dev_single = np.sqrt(np.maximum(pred_var_single, 1e-18))
                lower_95_single = pred_mean_single - 1.96 * pred_std_dev_single
                upper_95_single = pred_mean_single + 1.96 * pred_std_dev_single
                output_domain = np.linspace(0, 1, current_n_eta)
                plt.figure(figsize=(10, 6))
                plt.plot(output_domain, actual_single, 'k-', lw=2, label='Actual Simulator Output')
                plt.plot(output_domain, pred_mean_single, 'r--', lw=2, label='Emulator Mean Prediction')
                plt.fill_between(output_domain, lower_95_single, upper_95_single, color='red', alpha=0.2, label='Emulator 95% CI')
                plt.title(f"Emulator vs Actual Output {plot_title_suffix} - Test Point {test_point_to_plot_recon}", fontsize=14)
                plt.xlabel("Output Dimension Index / Domain", fontsize=12)
                plt.ylabel("Output Value", fontsize=12)
                plt.legend(fontsize=10)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                plt.show()


        # === PLOT TYPE 4: PCA Variance Explained (Multivariate Only) ===
        if current_n_eta > 1 and pca_obj and plot_pca_variance: # Added flag check
            print("\nGenerating PCA variance explained plot...")
                # (Plotting code remains the same as before)
            plt.figure(figsize=(6, 4))
            n_components_actual = pca_obj.n_components_
            plt.bar(range(1, n_components_actual + 1), pca_obj.explained_variance_ratio_)
            plt.plot(range(1, n_components_actual + 1), np.cumsum(pca_obj.explained_variance_ratio_), 'r-o', label='Cumulative Variance')
            plt.xlabel("Principal Component")
            plt.ylabel("Variance Explained Ratio")
            plt.title("PCA Explained Variance by Component")
            plt.ylim(0, 1.05)
            plt.xticks(range(1, n_components_actual + 1))
            plt.legend()
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.show()


    print("\n--- Workflow Complete ---")