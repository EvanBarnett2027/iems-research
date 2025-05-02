import numpy as np
import tensorflow as tf
import gpflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(123)
tf.random.set_seed(123)

# --- Configuration ---
gpflow.config.set_default_float(tf.float64)
gpflow.config.set_default_summary_fmt("notebook")

# --- Helper Functions ---

def run_simulator(inputs):
    """
    A placeholder for a complex, high-dimensional simulator.
    This example uses a synthetic function based on Branin and sine waves.
    Output dimension n_eta = 50.
    Input dimension depends on the shape of 'inputs'.
    """
    n_sim, input_dim = inputs.shape
    n_eta = 50 # Dimension of the simulator output vector
    t = np.linspace(0, 1, n_eta)
    outputs = np.zeros((n_sim, n_eta))

    # Example: Use Branin function logic (scaled) for amplitude
    # and sine waves based on inputs for shape. Assumes input_dim >= 2.
    x1 = inputs[:, 0] * 15 - 5  # Scale input 1 to [-5, 10]
    x2 = inputs[:, 1] * 15      # Scale input 2 to [0, 15]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - 1.0 / (8.0 * np.pi)) * np.cos(x1)
    branin_val = (term1 + term2 + s) / 50 # Scaled Branin

    for i in range(n_sim):
        freq = 1.0 + inputs[i, 0] * 5 # Frequency depends on input 1
        phase = inputs[i, 1] * np.pi # Phase depends on input 2
        noise = np.random.randn(n_eta) * 0.01 # Small simulator noise

        # Add more input dependencies if input_dim > 2
        amplitude_mod = 1.0
        if input_dim > 2:
           amplitude_mod += inputs[i, 2] * 0.5 # Input 3 modulates amplitude

        outputs[i, :] = (branin_val[i] * amplitude_mod * np.sin(2 * np.pi * freq * t + phase)) + noise

    return outputs


def train_pcgp_simulator(sim_inputs, sim_outputs, num_pcs):
    """
    Trains the PCGP model for the simulator.

    Args:
        sim_inputs (np.ndarray): Simulator input settings (m x d).
                                 Assumed to be scaled (e.g., to [0, 1]).
        sim_outputs (np.ndarray): Corresponding high-dimensional simulator
                                  outputs (m x n_eta).
        num_pcs (int): Number of principal components (p_eta) to use.

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
    print(f"  Principal Components (p_eta): {num_pcs}")

    # 1. Scale Inputs (Good practice for GPs)
    input_scaler = StandardScaler()
    sim_inputs_scaled = input_scaler.fit_transform(sim_inputs)

    # 2. Scale & Center Outputs (Standard practice for PCA)
    output_scaler = StandardScaler(with_std=True) # Center and scale variance
    sim_outputs_scaled = output_scaler.fit_transform(sim_outputs)


    # 3. Perform PCA
    pca = PCA(n_components=num_pcs, svd_solver='full')
    # Fit PCA on scaled outputs and get the weights (scores)
    # weights shape: (m, num_pcs)
    weights = pca.fit_transform(sim_outputs_scaled)
    print(f"  PCA Variance Explained by {num_pcs} components: {np.sum(pca.explained_variance_ratio_):.4f}")


    # 4. Train Independent GPs for each PC weight
    gp_models = []
    for i in range(num_pcs):
        print(f"  Training GP for PC {i+1}/{num_pcs}...")
        # Use inputs as X, and the i-th column of weights as Y
        Y_pc = weights[:, i:i+1] # Target for this GP model

        # Define the GP model (GPR - exact inference)
        # Kernel: RBF (SquaredExponential) with ARD (Automatic Relevance Determination)
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=[1.0] * input_dim, # Initialize lengthscales (ARD)
            variance=1.0
        )
        # Set priors (optional, but recommended for stability)
        # kernel.variance.prior = tfp.distributions.LogNormal(np.float64(0.0), np.float64(1.0))
        # kernel.lengthscales.prior = tfp.distributions.LogNormal(np.float64(0.0), np.float64(1.0))


        # Use zero mean function as data is centered by PCA / StandardScaler
        model = gpflow.models.GPR(data=(sim_inputs_scaled, Y_pc),
                                   kernel=kernel,
                                   mean_function=None)

        # Set trainable parameters (likelihood variance might be fixed or trained)
        gpflow.set_trainable(model.likelihood.variance, True)
        # Optional: Add prior to likelihood variance
        # model.likelihood.variance.prior = tfp.distributions.LogNormal(np.float64(-2.0), np.float64(1.0))


        # Optimize the model hyperparameters
        optimizer = gpflow.optimizers.Scipy()
        try:
             opt_log = optimizer.minimize(model.training_loss,
                                     model.trainable_variables,
                                     options=dict(maxiter=500, disp=False)) # Increase maxiter if needed
             # print(f"    Optimization success: {opt_log.success}")
             # print(f"    Final Loss: {opt_log.fun:.4f}")
             # print(f"    Kernel Variance: {model.kernel.variance.numpy():.4f}")
             # print(f"    Kernel Lengthscales: {model.kernel.lengthscales.numpy()}")
             # print(f"    Likelihood Variance: {model.likelihood.variance.numpy():.4f}")

        except Exception as e:
            print(f"    Optimization failed for PC {i+1}: {e}")
            # Handle failure, e.g., use initial hyperparameters or skip this PC


        gp_models.append(model)

    return gp_models, pca, input_scaler, output_scaler


def predict_pcgp_simulator(new_inputs, gp_models, pca, input_scaler, output_scaler):
    """
    Predicts the high-dimensional simulator output at new input locations.

    Args:
        new_inputs (np.ndarray): New input locations (n_new x d).
        gp_models (list): List of trained GPFlow GPR models.
        pca (PCA): Fitted sklearn PCA object from training.
        input_scaler (StandardScaler): Fitted input scaler from training.
        output_scaler (StandardScaler): Fitted output scaler from training.


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
    n_eta = pca.n_features_in_

    # 1. Scale new inputs
    new_inputs_scaled = input_scaler.transform(new_inputs)

    # 2. Predict weights (mean and variance) for each PC
    pred_weights_mean = np.zeros((n_new, num_pcs))
    pred_weights_var = np.zeros((n_new, num_pcs))

    for i, model in enumerate(gp_models):
        # predict_y returns mean and variance of Y (the weights)
        mean_w, var_w = model.predict_y(new_inputs_scaled)
        pred_weights_mean[:, i] = mean_w.numpy().flatten()
        pred_weights_var[:, i] = var_w.numpy().flatten()

        # Ensure variance is non-negative (numerical stability)
        pred_weights_var[:, i] = np.maximum(pred_weights_var[:, i], 1e-9)


    # 3. Reconstruct the high-dimensional output in the SCALED space
    # Mean: Use inverse_transform for convenience (handles mean addition)
    pred_mean_scaled = pca.inverse_transform(pred_weights_mean)

    # Variance: Combine variances using the basis vectors
    # Var(Y) = Var(sum(k_i * w_i)) = sum(k_i^2 * Var(w_i)) assuming independence
    # pca.components_ has shape (num_pcs, n_eta)
    basis_vectors_sq = pca.components_**2 # (p_eta, n_eta)
    # We need to multiply pred_weights_var (n_new, p_eta) with basis_vectors_sq
    # Result shape should be (n_new, n_eta)
    pred_var_scaled = pred_weights_var @ basis_vectors_sq # Matrix multiplication


    # 4. Transform back to the original output scale
    # Mean: Use the fitted output_scaler
    pred_mean_orig_scale = output_scaler.inverse_transform(pred_mean_scaled)

    # Variance: Must be scaled by the square of the output scale factor
    # var(s*Y) = s^2 * var(Y). output_scaler.scale_ contains the std dev (s).
    output_variance_scaling = output_scaler.scale_**2
    pred_var_orig_scale = pred_var_scaled * output_variance_scaling

    return pred_mean_orig_scale, pred_var_orig_scale


# --- Example Usage ---

# 1. Define Simulation Design
m = 50  # Number of simulation runs for training
input_dim = 3 # Number of input parameters (x, t)
sim_inputs_train = np.random.rand(m, input_dim) # Inputs in [0, 1]

# 2. Run the "Expensive" Simulator
sim_outputs_train = run_simulator(sim_inputs_train)
n_eta = sim_outputs_train.shape[1]

# 3. Train the PCGP Emulator
num_pcs = 5 # Choose the number of principal components (p_eta)
gp_models, pca, input_scaler, output_scaler = train_pcgp_simulator(
    sim_inputs_train, sim_outputs_train, num_pcs
)

# 4. Define New Inputs for Prediction
n_test = 10
test_inputs = np.random.rand(n_test, input_dim)

# 5. Make Predictions using the Emulator
pred_mean, pred_var = predict_pcgp_simulator(
    test_inputs, gp_models, pca, input_scaler, output_scaler
)

# 6. Compare Emulator Prediction to Actual Simulator Run (for validation)
actual_outputs_test = run_simulator(test_inputs)

# 7. Plot comparison for one test point
test_idx = 0
plt.figure(figsize=(12, 6))
output_domain = np.linspace(0, 1, n_eta)

# Plot actual simulator output
plt.plot(output_domain, actual_outputs_test[test_idx, :], 'k-', lw=2, label='Actual Simulator Output')

# Plot emulator prediction mean
plt.plot(output_domain, pred_mean[test_idx, :], 'r--', lw=2, label='Emulator Mean Prediction')

# Plot uncertainty bounds (e.g., +/- 2 standard deviations)
std_dev = np.sqrt(pred_var[test_idx, :])
plt.fill_between(output_domain,
                 pred_mean[test_idx, :] - 1.96 * std_dev,
                 pred_mean[test_idx, :] + 1.96 * std_dev,
                 color='red', alpha=0.2, label='Emulator 95% CI')

plt.title(f"Emulator Prediction vs Actual Simulator Output (Test Point {test_idx})")
plt.xlabel("Output Dimension Index (e.g., Time or Space)")
plt.ylabel("Simulator Output Value")
plt.legend()
plt.grid(True)
plt.show()

# Print some stats for the test point
rmse = np.sqrt(np.mean((actual_outputs_test[test_idx, :] - pred_mean[test_idx, :])**2))
print(f"\nRMSE for test point {test_idx}: {rmse:.4f}")

avg_std_dev = np.mean(std_dev)
print(f"Average predicted standard deviation for test point {test_idx}: {avg_std_dev:.4f}")

# --- Optional: Plot Explained Variance ---
plt.figure(figsize=(6, 4))
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), 'r-o')
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained Ratio")
plt.title("PCA Explained Variance by Component")
plt.grid(True, axis='y')
plt.show()