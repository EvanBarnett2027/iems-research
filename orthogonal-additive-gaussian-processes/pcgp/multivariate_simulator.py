# multivariate_simulator.py
"""Multivariate output simulator example."""
import numpy as np

# Define the metadata dictionary required by main.py
_dict = {
    'function': 'MultiSine', # Descriptive name
    'xdim': 2,              # e.g., Amplitude, Phase controls
    'thetadim': 2,          # e.g., Frequency, Offset controls
    'n_eta_out': 50,        # Explicitly state output dimension
    # Input names for potential use in plotting labels
    'input_names_x': ['Amplitude Control', 'Phase Control'],
    'input_names_theta': ['Frequency Control', 'Offset Control']
}

# --- Core Simulation Logic ---

def multivariate_model_vec(x_std, theta_std, t_domain):
    """
    Calculates a vector output based on varying sine wave properties.

    Args:
        x_std (np.ndarray): Standardized x inputs, shape (m, xdim).
                            Assumed order: [amp_ctrl, phase_ctrl].
        theta_std (np.ndarray): Standardized theta inputs, shape (m, thetadim).
                                Assumed order: [freq_ctrl, offset_ctrl].
        t_domain (np.ndarray): The domain over which the output curve is defined, shape (n_eta,).

    Returns:
        np.ndarray: The calculated output curves, shape (m, n_eta).
    """
    m = x_std.shape[0]
    n_eta = t_domain.shape[0]

    # Ensure inputs are 2D
    if x_std.ndim == 1: x_std = np.atleast_2d(x_std)
    if theta_std.ndim == 1: theta_std = np.atleast_2d(theta_std)

    # Extract standardized controls (shape will be (m, 1))
    amp_ctrl_std    = x_std[:, 0:1]
    phase_ctrl_std  = x_std[:, 1:2]
    freq_ctrl_std   = theta_std[:, 0:1]
    offset_ctrl_std = theta_std[:, 1:2]

    # --- Map standardized controls [0,1] to 'physical' parameters ---
    # Adjust these ranges as needed for your desired behavior
    min_amp, max_amp = 0.5, 2.5
    min_phase, max_phase = 0, np.pi
    min_freq, max_freq = 1.0, 5.0
    min_offset, max_offset = -1.0, 1.0

    amplitude = amp_ctrl_std * (max_amp - min_amp) + min_amp
    phase     = phase_ctrl_std * (max_phase - min_phase) + min_phase
    frequency = freq_ctrl_std * (max_freq - min_freq) + min_freq
    offset    = offset_ctrl_std * (max_offset - min_offset) + min_offset
    # --- End Parameter Mapping ---

    # Reshape t_domain for broadcasting: (1, n_eta)
    t_row = t_domain.reshape(1, -1)

    # Calculate the sine wave output using broadcasting
    # amplitude (m, 1), frequency (m, 1), t_row (1, n_eta), phase (m, 1), offset (m, 1)
    # The calculation inside sin() broadcasts to (m, n_eta)
    # The final result broadcasts to (m, n_eta)
    output_curves = amplitude * np.sin(2 * np.pi * frequency * t_row + phase) + offset

    # Optional: Add a small amount of simulated "noise" if desired
    # noise_level = 0.05
    # output_curves += np.random.normal(0, noise_level, size=(m, n_eta))

    return output_curves


# --- Wrapper Function for main.py ---

def run_simulator_multivariate(combined_inputs_std):
    """
    Runs the multivariate simulator for each row of combined_inputs_std.

    Args:
        combined_inputs_std (np.ndarray): Array of shape (m, d) where d = xdim + thetadim.
                                          Each row contains standardized x and theta values
                                          concatenated. Assumed order: [x_std..., theta_std...].
                                          Inputs MUST be in the range [0, 1].

    Returns:
        np.ndarray: Array of shape (m, n_eta) containing the multivariate output
                    for each input row.
    """
    m, total_dim = combined_inputs_std.shape
    xdim = _dict['xdim']
    thetadim = _dict['thetadim']
    n_eta = _dict['n_eta_out'] # Get output dimension from metadata

    if total_dim != (xdim + thetadim):
        raise ValueError(f"Input dimension {total_dim} != xdim+thetadim ({xdim}+{thetadim})")

    # Define the output domain (e.g., time or space)
    output_domain = np.linspace(0, 1, n_eta)

    # Extract standardized x and theta for all runs
    x_std_batch = combined_inputs_std[:, :xdim]      # Shape (m, xdim)
    theta_std_batch = combined_inputs_std[:, xdim:]  # Shape (m, thetadim)

    # Call the core vectorized function
    outputs = multivariate_model_vec(x_std_batch, theta_std_batch, output_domain)

    # Shape should already be (m, n_eta)
    return outputs


# --- Optional: Self-Test Block ---
if __name__ == "__main__":
    print(f"Running self-test for {__name__}")

    num_samples = 5
    xdim_test = _dict['xdim']
    thetadim_test = _dict['thetadim']
    n_eta_test = _dict['n_eta_out']
    combined_inputs_test = np.random.rand(num_samples, xdim_test + thetadim_test)

    print(f"\nTesting run_simulator_multivariate with {num_samples} samples...")
    simulator_outputs = run_simulator_multivariate(combined_inputs_test)
    print(f"Input shape: {combined_inputs_test.shape}")
    print(f"Output shape: {simulator_outputs.shape}")

    # Verify dimension consistency
    assert simulator_outputs.shape == (num_samples, n_eta_test)
    print("Output shape test PASSED.")

    # Optionally plot one sample output curve
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.linspace(0, 1, n_eta_test), simulator_outputs[0, :], label="Sample Output Curve 0")
    plt.title("Sample Output from Multivariate Simulator")
    plt.xlabel("Output Dimension Index")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nSelf-test complete.")