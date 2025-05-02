import numpy as np
import tensorflow as tf
import gpflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import from our simulator file
from wingweight_simulator import run_simulator_wingweight, _dict as ww_dict # Renaming _dict to avoid confusion

# --- Configuration ---
gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_summary_fmt("notebook")
np.random.seed(42) # Using a different seed for this script if desired
tf.random.set_seed(42)


# --- PCGP Helper Functions ---
def train_pcgp_simulator(sim_inputs, sim_outputs, num_pcs=None):
    """
    Trains the PCGP model for the simulator. Handles scalar output gracefully.
    (Code for this function is the same as in the previous combined script)
    
    Args:
        sim_inputs (np.ndarray): Input points (design locations, x and theta) where
                                 the simulator was run.
                                 Assumed to be scaled to [0, 1].
                                 Shape (m_train, total_input_dim)
        sim_outputs (np.ndarray): Corresponding high-dimensional simulator
                                  outputs.
                                  Shape (m_train, n_eta), where n_eta
                                  is the output diemsion.
        num_pcs (int): Number of principal components (p_eta) to use.
                       If None, it is determined via the number needed
                       to meet a variance threshold.

        Returns:
        tuple: (list_of_gp_models, pca_transformer, input_scaler, output_scaler)
               - list_of_gp_models: List of trained GPFlow GPR models (one per PC).
               - pca_transformer: Fitted sklearn PCA object.
               - input_scaler: Fitted sklearn StandardScaler for inputs.
               - output_scaler: Fitted sklearn StandardScaler for outputs (applied before PCA).
    """
    
    m, input_dim = sim_inputs.shape
    _, n_eta = sim_outputs.shape

    print(f"Training PCGP Simulator:")
    print(f"  Simulations (m): {m}")
    print(f"  Input dim (d): {input_dim}")
    print(f"  Output dim (n_eta): {n_eta}")

    input_scaler = StandardScaler()
    sim_inputs_scaled = input_scaler.fit_transform(sim_inputs)

    output_scaler = StandardScaler(with_std=True)
    sim_outputs_scaled = output_scaler.fit_transform(sim_outputs)

    weights = sim_outputs_scaled
    pca = None

    if n_eta > 1:
        if num_pcs is None:
            pca_full = PCA(svd_solver='full')
            pca_full.fit(sim_outputs_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            num_pcs = np.argmax(cumsum_var >= 0.999) + 1
            print(f"  Dynamically selected num_pcs = {num_pcs} to explain >= 99.9% variance.")
        elif num_pcs > n_eta:
             print(f"  Warning: num_pcs ({num_pcs}) > n_eta ({n_eta}). Setting num_pcs=n_eta.")
             num_pcs = n_eta

        pca = PCA(n_components=num_pcs, svd_solver='full')
        weights = pca.fit_transform(sim_outputs_scaled)
        print(f"  PCA Variance Explained by {num_pcs} components: {np.sum(pca.explained_variance_ratio_):.4f}")
    else:
        num_pcs = 1
        print("  Scalar output detected. Skipping PCA, training single GP.")

    gp_models = []
    for i in range(num_pcs):
        print(f"  Training GP for Component {i+1}/{num_pcs}...")
        Y_pc = weights[:, i:i+1]
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * input_dim, variance=1.0)
        model = gpflow.models.GPR(data=(sim_inputs_scaled, Y_pc), kernel=kernel, mean_function=None)
        gpflow.set_trainable(model.likelihood.variance, True)
        optimizer = gpflow.optimizers.Scipy()
        try:
             opt_log = optimizer.minimize(model.training_loss,
                                     model.trainable_variables,
                                     options=dict(maxiter=500, disp=False))
             if not opt_log.success:
                  print(f"    Warning: Optimization may not have fully converged for Component {i+1}")
        except Exception as e:
            print(f"    ERROR: Optimization failed for Component {i+1}: {e}")
        gp_models.append(model)
    return gp_models, pca, input_scaler, output_scaler


def predict_pcgp_simulator(new_inputs, gp_models, pca, input_scaler, output_scaler):
    """
    Predicts the simulator output at new input locations. Handles scalar output.
    (Code for this function is the same as in the previous combined script)
    
    Args:
        new_inputs: The new input points where predictions are desired. Assumed to be standardized [0, 1]. Shape (m_test, total_input_dim).
        
        gp_models: The list of trained GP models from train_pcgp_simulator.
        
        pca: The fitted PCA object (or None) from train_pcgp_simulator.
        
        input_scaler: The fitted input scaler object.
        
        output_scaler: The fitted output scaler object.


    Returns:
        tuple: (pred_mean_orig_scale, pred_var_orig_scale)
               - pred_mean_orig_scale: Predicted mean output (n_new x n_eta)
                                       in the original simulator output scale.
               - pred_var_orig_scale: Predicted variance for each output dimension
                                      (n_new x n_eta) in the original scale squared.
                                      This assumes independence of PC predictions
                                      for variance calculation simplicity.
    
    """
    n_new, input_dim = new_inputs.shape
    num_pcs = len(gp_models)

    if pca is not None:
        n_eta = pca.n_features_in_
    else:
        n_eta = 1 # Scalar output

    new_inputs_scaled = input_scaler.transform(new_inputs)

    pred_weights_mean = np.zeros((n_new, num_pcs))
    pred_weights_var = np.zeros((n_new, num_pcs))

    for i, model in enumerate(gp_models):
        mean_w, var_w = model.predict_y(new_inputs_scaled)
        pred_weights_mean[:, i] = mean_w.numpy().flatten()
        pred_weights_var[:, i] = var_w.numpy().flatten()
        pred_weights_var[:, i] = np.maximum(pred_weights_var[:, i], 1e-9)

    if pca is not None and n_eta > 1:
        pred_mean_scaled = pca.inverse_transform(pred_weights_mean)
        basis_vectors_sq = pca.components_**2
        pred_var_scaled = pred_weights_var @ basis_vectors_sq
    else:
        pred_mean_scaled = pred_weights_mean
        pred_var_scaled = pred_weights_var

    pred_mean_orig_scale = output_scaler.inverse_transform(pred_mean_scaled)
    output_variance_scaling = output_scaler.scale_**2
    pred_var_orig_scale = pred_var_scaled * output_variance_scaling
    return pred_mean_orig_scale, pred_var_orig_scale


# --- Main Script ---
if __name__ == "__main__":
    # 1. Define Simulation Design
    m_train = 100  # Number of simulation runs for training
    xdim = ww_dict['xdim']
    thetadim = ww_dict['thetadim']
    total_input_dim = xdim + thetadim

    # Generate standardized inputs in [0, 1]
    # Using simple uniform sampling for now
    sim_inputs_train_std = np.random.rand(m_train, total_input_dim)

    # 2. Run the Wingweight Simulator
    print("Running Wingweight simulator for training data...")
    # Pass the standardized inputs to the simulator function
    sim_outputs_train = run_simulator_wingweight(sim_inputs_train_std)
    print(f"Simulator training output shape: {sim_outputs_train.shape}")

    # 3. Train the PCGP Emulator
    # num_pcs is ignored for scalar output, but we pass None for consistency
    gp_models, pca_obj, input_scaler_obj, output_scaler_obj = train_pcgp_simulator(
        sim_inputs_train_std, sim_outputs_train, num_pcs=None
    )

    # 4. Define New Inputs for Prediction (Standardized [0, 1])
    m_test = 20
    test_inputs_std = np.random.rand(m_test, total_input_dim)

    # 5. Make Predictions using the Emulator
    print("\nMaking predictions on test data...")
    pred_mean, pred_var = predict_pcgp_simulator(
        test_inputs_std, gp_models, pca_obj, input_scaler_obj, output_scaler_obj
    )

    # 6. Compare Emulator Prediction to Actual Simulator Run (for validation)
    print("Running simulator for test points for validation...")
    actual_outputs_test = run_simulator_wingweight(test_inputs_std)

    # 7. Plot comparison (Scatter plot for scalar output)
    plt.figure(figsize=(8, 8))
    min_val = min(actual_outputs_test.min(), pred_mean.min()) * 0.95
    max_val = max(actual_outputs_test.max(), pred_mean.max()) * 1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal (y=x)')

    std_dev = np.sqrt(pred_var)
    plt.errorbar(actual_outputs_test.flatten(), pred_mean.flatten(), yerr=1.96 * std_dev.flatten(),
                 fmt='o', color='dodgerblue', ecolor='lightskyblue', elinewidth=2, capsize=3, label='Emulator Prediction (95% CI)')

    plt.title("Emulator Prediction vs Actual Simulator Output (Wingweight)")
    plt.xlabel("Actual Simulator Output (Wing Weight)")
    plt.ylabel("Emulator Mean Prediction (Wing Weight)")
    plt.legend()
    plt.grid(True)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # plt.axis('equal') # Sometimes makes plot too small if ranges differ a lot
    plt.show()

    # --- NEW: Plot 2: Prediction vs Single Input (Smooth) ---
    print("\nGenerating plot of Emulator Prediction vs. a single input...")

    # --- Configuration for the smooth plot ---
    var_idx_to_plot = 0 # Index of the input variable to plot (0 to total_input_dim-1)
                        # 0 corresponds to xstd_1 (Sw)
                        # 6 corresponds to tstd_1 (A)
    n_plot_points = 101 # Number of points for the dense grid
    fixed_val = 0.5     # Standardized value for other inputs
    # --- End Configuration ---

    if var_idx_to_plot >= total_input_dim:
        print(f"Error: var_idx_to_plot ({var_idx_to_plot}) is out of bounds (max {total_input_dim-1}).")
    else:
        # Create the dense input grid for plotting
        plot_inputs_std = np.full((n_plot_points, total_input_dim), fixed_val)
        varying_input_std = np.linspace(0, 1, n_plot_points) # Standardized range [0,1]
        plot_inputs_std[:, var_idx_to_plot] = varying_input_std

        # Make predictions across the grid using the emulator
        plot_pred_mean, plot_pred_var = predict_pcgp_simulator(
            plot_inputs_std, gp_models, pca_obj, input_scaler_obj, output_scaler_obj
        )

        # Calculate confidence intervals
        plot_std_dev = np.sqrt(plot_pred_var)
        plot_lower_95 = plot_pred_mean - 1.96 * plot_std_dev
        plot_upper_95 = plot_pred_mean + 1.96 * plot_std_dev

        # Create the plot
        plt.figure(figsize=(8, 5))

        # Plot predictive means as blue line
        plt.plot(varying_input_std, plot_pred_mean.flatten(), 'b', lw=2, label='Emulator Mean')

        # Shade between the lower and upper confidence bounds
        plt.fill_between(varying_input_std, plot_lower_95.flatten(), plot_upper_95.flatten(),
                         color='lightblue', alpha=0.5, label='Emulator 95% CI')

        # Optional: Plot training data points *that are relevant*
        # This is a bit tricky as we only vary one dimension. We can show
        # the training points projected onto this dimension.
        # For clarity, let's only plot the mean and CI for now.
        # You could add: 
        plt.plot(sim_inputs_train_std[:, var_idx_to_plot], sim_outputs_train, 'bo', alpha=0.1, label='Training Data (projection)')


        # Determine appropriate labels based on index
        if var_idx_to_plot < xdim:
            input_var_name = f'x_std[{var_idx_to_plot+1}]'
            # You could map this back to physical names like 'Sw', 'Wfw', etc. if desired
        else:
            input_var_name = f'theta_std[{var_idx_to_plot - xdim + 1}]'
            # Map back to 'A', 'LamCaps', etc.

        plt.title(f"Emulator Prediction vs. Standardized Input '{input_var_name}'", fontsize=14)
        plt.xlabel(f"Standardized Input '{input_var_name}' (Others Fixed at {fixed_val})", fontsize=12)
        plt.ylabel("Emulator Predicted Output (Wing Weight)", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()
    # --- End of Plot 2 ---

    # Print some overall stats
    rmse = np.sqrt(np.mean((actual_outputs_test - pred_mean)**2))
    print(f"\nOverall RMSE on test set: {rmse:.4g}")
    mean_abs_err = np.mean(np.abs(actual_outputs_test - pred_mean))
    print(f"Mean Absolute Error on test set: {mean_abs_err:.4g}")
    avg_pred_std_dev = np.mean(std_dev)
    print(f"Average Predicted Standard Deviation on test set: {avg_pred_std_dev:.4g}")

    if pca_obj: # Only try to plot if PCA was actually done
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, pca_obj.n_components_ + 1), pca_obj.explained_variance_ratio_)
        plt.plot(range(1, pca_obj.n_components_ + 1), np.cumsum(pca_obj.explained_variance_ratio_), 'r-o')
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained Ratio")
        plt.title("PCA Explained Variance by Component")
        plt.grid(True, axis='y')
        plt.show()
    else:
        print("\nPCA not performed (scalar output).")