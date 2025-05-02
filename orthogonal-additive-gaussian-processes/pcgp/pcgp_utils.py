# pcgp_utils.py
import numpy as np
import tensorflow as tf
import gpflow
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# Conditional OAK Import - assuming OAK is installed or accessible
try:
    from oak.model_utils import oak_model
    OAK_AVAILABLE = True
except ImportError:
    OAK_AVAILABLE = False
    print("Warning: OAK module not found. OAK model type will not be available.")


def train_emulator(
    sim_inputs_std,
    sim_outputs,
    model_type='StandardGP',
    num_pcs=None,
    oak_config=None
):
    """
    Trains an emulator model (Standard GP with PCA, or OAK with PCA for multivariate).

    Args:
        sim_inputs_std (np.ndarray): Simulator input settings (m x d), standardized [0, 1].
        sim_outputs (np.ndarray): Corresponding simulator outputs (m x n_eta).
        model_type (str): 'StandardGP' or 'OAK'.
        num_pcs (int, optional): Number of PCs for StandardGP/OAK multivariate.
                                 If None, uses variance threshold for PCA.
        oak_config (dict, optional): Configuration dictionary for OAK model init.

    Returns:
        tuple: Depending on model_type:
            StandardGP: (gp_models_list, pca_obj, input_scaler, output_scaler)
            OAK: (oak_models_list, pca_obj, input_scaler, output_scaler)
            Returns (None, None, None, None) on failure.
    """
    m, input_dim = sim_inputs_std.shape
    if sim_outputs.ndim == 1:
        sim_outputs = sim_outputs.reshape(-1, 1)
    _, n_eta = sim_outputs.shape

    print(f"\n--- Training Emulator (Type: {model_type}) ---")
    print(f"Simulations (m): {m}")
    print(f"Input dim (d): {input_dim}")
    print(f"Output dim (n_eta): {n_eta}")

    # --- Step 1: Scale Inputs and Outputs ---
    # Input scaler (even for std inputs, helps internal GP optimization)
    input_scaler = StandardScaler()
    sim_inputs_scaled_for_gp = input_scaler.fit_transform(sim_inputs_std) # Used by StandardGP GPR

    # Output scaler (applied before PCA)
    output_scaler = StandardScaler(with_std=True)
    sim_outputs_scaled = output_scaler.fit_transform(sim_outputs)

    # --- Step 2: PCA (if n_eta > 1) ---
    pca_obj = None
    weights = sim_outputs_scaled # Target for emulation is scaled outputs if n_eta=1
    actual_num_pcs = 1

    if n_eta > 1:
        print("Performing PCA on scaled outputs...")
        if num_pcs is None:
            pca_full = PCA(svd_solver='full')
            pca_full.fit(sim_outputs_scaled)
            cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
            actual_num_pcs = np.argmax(cumsum_var >= 0.999) + 1
            print(f"Dynamically selected num_pcs = {actual_num_pcs} to explain >= 99.9% variance.")
        elif num_pcs > n_eta:
             print(f"Warning: num_pcs ({num_pcs}) > n_eta ({n_eta}). Setting num_pcs=n_eta.")
             actual_num_pcs = n_eta
        else:
             actual_num_pcs = num_pcs

        pca_obj = PCA(n_components=actual_num_pcs, svd_solver='full')
        weights = pca_obj.fit_transform(sim_outputs_scaled) # Shape: (m, actual_num_pcs)
        print(f"PCA Variance Explained by {actual_num_pcs} components: {np.sum(pca_obj.explained_variance_ratio_):.4f}")
        print(f"Target for emulation: {actual_num_pcs} PCA weights.")
    else:
        print("Scalar output. Target for emulation: Scaled output.")

    # --- Step 3: Train Emulator Model(s) for each weight/output ---
    emulator_models = [] # List to store trained GPflow GPR or OAK instances

    if model_type == 'StandardGP':
        print(f"Training {actual_num_pcs} Standard GP model(s)...")
        for i in range(actual_num_pcs):
            Y_target = weights[:, i:i+1] # Target is the i-th weight (or scaled output)
            kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0] * input_dim, variance=1.0)
            # Use SCALED inputs for standard GP GPR model
            model = gpflow.models.GPR(data=(sim_inputs_scaled_for_gp, Y_target), kernel=kernel, mean_function=None)
            gpflow.set_trainable(model.likelihood.variance, True)
            optimizer = gpflow.optimizers.Scipy()
            try:
                opt_log = optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=500, disp=False))
            except Exception as e:
                print(f"    ERROR: Optimization failed for Component {i+1}: {e}")
                return None, None, None, None # Failure
            emulator_models.append(model)

    elif model_type == 'OAK':
        if not OAK_AVAILABLE:
            print("Error: OAK model type selected, but OAK module is not available.")
            return None, None, None, None

        if oak_config is None:
            print("Warning: OAK model selected but no oak_config provided. Using defaults.")
            oak_config = {}

        print(f"Training {actual_num_pcs} OAK model(s)...")
        for i in range(actual_num_pcs):
            print(f"  Training OAK for Component {i+1}/{actual_num_pcs}...")
            Y_target = weights[:, i:i+1] # Target is the i-th weight

            oak_instance = oak_model(
                max_interaction_depth=oak_config.get('max_interaction_depth', input_dim),
                num_inducing=oak_config.get('num_inducing', 200),
                use_sparsity_prior=oak_config.get('use_sparsity_prior', True),
                use_normalising_flow=oak_config.get('use_normalising_flow', False),
                share_var_across_orders=oak_config.get('share_var_across_orders', True),
                sparse=oak_config.get('sparse', False) or m > 1000,
            )
            t_start = time.time()
            try:
                # Pass standardized inputs [0,1] and the specific PCA weight vector
                oak_instance.fit(sim_inputs_std, Y_target, optimise=True)
                print(f"    OAK fitting took {time.time() - t_start:.1f} seconds.")
            except Exception as e:
                print(f"    ERROR during OAK fitting for Component {i+1}: {e}")
                return None, None, None, None # Failure
            emulator_models.append(oak_instance) # Store the whole OAK instance

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"--- Emulator Training Complete (Type: {model_type}) ---")
    return emulator_models, pca_obj, input_scaler, output_scaler


def predict_emulator(
    new_inputs_std,
    emulator_models, # List of trained models (GPR or OAK instances)
    model_type,
    pca_obj,         # PCA object (None if n_eta=1)
    input_scaler,    # Scaler fitted on training inputs
    output_scaler    # Scaler fitted on training outputs
):
    """
    Predicts simulator output using the trained emulator.

    Args:
        new_inputs_std (np.ndarray): New input locations (n_new x d), standardized [0, 1].
        emulator_models (list): List of trained GPFlow GPR or OAK models.
        model_type (str): 'StandardGP' or 'OAK'.
        pca_obj (PCA or None): Fitted PCA object from training.
        input_scaler (StandardScaler): Fitted input scaler from training.
        output_scaler (StandardScaler): Fitted output scaler from training.

    Returns:
        tuple: (pred_mean_orig_scale, pred_var_orig_scale) or (None, None) on failure
    """
    print(f"\n--- Predicting with Emulator (Type: {model_type}) ---")
    if not emulator_models:
        print("Error: No trained emulator models provided.")
        return None, None

    n_new, input_dim = new_inputs_std.shape
    num_components = len(emulator_models) # Number of GPs or OAK models trained

    # Determine output dimension n_eta
    if pca_obj is not None:
        n_eta = pca_obj.n_features_in_
    else:
        n_eta = 1 # Must be scalar if no PCA

    # --- Step 1: Predict Weights/Scaled Output for each component ---
    pred_weights_mean = np.zeros((n_new, num_components))
    pred_weights_var = np.zeros((n_new, num_components))

    print(f"Predicting {num_components} component(s)...")
    if model_type == 'StandardGP':
        # Scale inputs for standard GPR prediction
        new_inputs_scaled_for_gp = input_scaler.transform(new_inputs_std)
        for i, model in enumerate(emulator_models):
            try:
                mean_w, var_w = model.predict_y(new_inputs_scaled_for_gp)
                pred_weights_mean[:, i] = mean_w.numpy().flatten()
                pred_weights_var[:, i] = var_w.numpy().flatten()
            except Exception as e:
                 print(f"Error predicting with Standard GP component {i}: {e}")
                 return None, None
    elif model_type == 'OAK':
        if not OAK_AVAILABLE:
            print("Error: OAK model type selected, but OAK module is not available.")
            return None, None
        for i, oak_instance in enumerate(emulator_models):
            # OAK's predict needs raw inputs to apply its internal scaling
            # But we need mean AND variance in the scaled space (like PCA weights)
            try:
                # a) Transform inputs using OAK's internal scaler for THIS component
                X_scaled_new = oak_instance._transform_x(new_inputs_std)
                # b) Predict latent mean/var using the underlying GPflow model
                mean_f, var_f = oak_instance.m.predict_f(X_scaled_new)
                # c) Get likelihood variance for THIS component
                likelihood_var = oak_instance.m.likelihood.variance.numpy()
                # d) Convert latent mean/var to predictive mean/var in the SCALED Y space
                #    (the PCA weight space) using THIS component's output scaler
                pred_mean_scaled_y = oak_instance.scaler_y.inverse_transform(mean_f.numpy())
                # Var(Y) = Var(scale*f + mean) + noise_var*scale^2 = scale^2*Var(f) + noise_var*scale^2
                var_f_scaled_y = var_f.numpy() * (oak_instance.scaler_y.scale_**2)
                pred_var_scaled_y = var_f_scaled_y + likelihood_var * (oak_instance.scaler_y.scale_**2)

                pred_weights_mean[:, i] = pred_mean_scaled_y.flatten()
                pred_weights_var[:, i] = pred_var_scaled_y.flatten()

            except Exception as e:
                 print(f"Error predicting with OAK component {i}: {e}")
                 return None, None
    else:
         raise ValueError(f"Unknown model_type: {model_type}")

    # Ensure variance is non-negative
    pred_weights_var = np.maximum(pred_weights_var, 1e-18)

    # --- Step 2: Reconstruct Output (if n_eta > 1) ---
    pred_mean_scaled_output = pred_weights_mean
    pred_var_scaled_output = pred_weights_var

    if pca_obj is not None: # Reconstruction needed
        print("Reconstructing full output from PCA weights...")
        # Mean: Use inverse_transform
        pred_mean_scaled_output = pca_obj.inverse_transform(pred_weights_mean)
        # Variance: Combine variances using the basis vectors
        basis_vectors_sq = pca_obj.components_**2 # (p_eta, n_eta)
        pred_var_scaled_output = pred_weights_var @ basis_vectors_sq # (n_new, n_eta)
    else: # Scalar output
        pass # Weights are already the scaled output

    # --- Step 3: Inverse Scale Output to Original Units ---
    pred_mean_orig_scale = output_scaler.inverse_transform(pred_mean_scaled_output)
    output_variance_scaling = output_scaler.scale_**2
    pred_var_orig_scale = pred_var_scaled_output * output_variance_scaling

    # Ensure final variance is non-negative
    pred_var_orig_scale = np.maximum(pred_var_orig_scale, 1e-18)

    print("--- Prediction Complete ---")
    return pred_mean_orig_scale, pred_var_orig_scale