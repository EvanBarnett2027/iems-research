# borehole_simulator.py
import numpy as np

# Define the metadata dictionary required by main.py
# Note: Physical bounds are implicitly defined within xstd2x and tstd2theta
_dict = {
    'function': 'Borehole',
    'xdim': 2,
    'thetadim': 4,
    # Add input names if desired for plotting labels later
    # 'input_names_x': ['rw', 'Hl'],
    # 'input_names_theta': ['Hu', 'Ld/Kw', 'T_eff', 'powparam']
    # Note: c_array and fail_array from the original code are related to
    # fail models, which are not used by the basic run_simulator,
    # so they are omitted here for clarity unless needed later.
}


# --- Helper Functions for Borehole Model ---

def tstd2theta(tstd, hard=True):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    # This function defines the physical ranges for theta parameters
    if tstd.ndim < 1.5:
        tstd = np.atleast_2d(tstd) # Ensure 2D for splitting

    # Assuming theta inputs are in order: T_eff, Hu, Ld/Kw, powparam
    # Make sure the order matches how combined_inputs_std is created/sliced
    Treffs = tstd[:, 0:1] # Slice to keep 2D shape
    Hus = tstd[:, 1:2]
    LdKws = tstd[:, 2:3]
    powparams = tstd[:, 3:4]

    # Apply transformations based on physical ranges
    Treff = (0.5 - 0.05) * Treffs + 0.05                     # Range [0.05, 0.5]
    Hu = Hus * (1110 - 990) + 990                            # Range [990, 1110]
    if hard: # Using the default 'hard=True' case
        Ld_Kw = LdKws * (1680 / 1500 - 1120 / 15000) + 1120 / 15000 # Range [7.47e-5, 1.12] approx
    else:
        Ld_Kw = LdKws * (1680 / 9855 - 1120 / 12045) + 1120 / 12045 # Range [9.30e-5, 0.17] approx

    powparam = powparams * (0.5 - (- 0.5)) + (-0.5)           # Range [-0.5, 0.5]

    # Combine in the order expected by borehole_vec: Hu, Ld_Kw, Treff, powparam
    theta_phys = np.hstack((Hu, Ld_Kw, Treff, powparam))
    return theta_phys


def xstd2x(xstd):
    """Given standardized x in [0, 1]^d, return non-standardized x."""
    # This function defines the physical ranges for x parameters
    if xstd.ndim < 1.5:
        xstd = np.atleast_2d(xstd) # Ensure 2D for splitting

    # Assuming x inputs are in order: rw, Hl
    rws_std = xstd[:, 0:1] # Slice to keep 2D shape
    Hls_std = xstd[:, 1:2]

    # Apply transformations based on physical ranges
    # rw is log-uniform between 0.05 and 0.5
    rw_log = rws_std * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw_log)                                      # Range [0.05, 0.5]
    Hl = Hls_std * (820 - 700) + 700                         # Range [700, 820]

    # Combine in the order expected by borehole_vec: rw, Hl
    x_phys = np.hstack((rw, Hl))
    return x_phys


def borehole_vec(x_phys, theta_phys):
    """Given PHYSICAL x and PHYSICAL theta, return vector of values."""
    # Order: Hu, Ld_Kw, Treff, powparam
    Hu = theta_phys[:, 0:1]
    Ld_Kw = theta_phys[:, 1:2]
    Treff = theta_phys[:, 2:3]
    powparam = theta_phys[:, 3:4]
    # Order: rw, Hl
    rw = x_phys[:, 0:1]
    Hl = x_phys[:, 1:2]

    # Calculate flow rate
    numer = 2 * np.pi * (Hu - Hl)
    denom1 = 2 * Ld_Kw / rw ** 2
    denom2 = Treff

    # Avoid division by zero or very small numbers if possible (though ranges should prevent it)
    denom = denom1 + denom2
    # Add small epsilon if denominator can be zero (check parameter ranges first)
    # epsilon = 1e-10
    # denom = np.where(np.abs(denom) < epsilon, np.sign(denom) * epsilon, denom)

    f = ((numer / denom) * np.exp(powparam * rw)).reshape(-1)
    return f


def borehole_model(x_std, theta_std):
    """
    Given standardized x_std and standardized theta_std, calculates the Borehole output.
    Designed to handle multiple x and theta inputs for potential cross-product evaluation.
    """
    # Convert standardized inputs to physical scales
    theta_phys = tstd2theta(theta_std)
    x_phys = xstd2x(x_std)

    # Ensure inputs are 2D
    if x_phys.ndim == 1: x_phys = np.atleast_2d(x_phys)
    if theta_phys.ndim == 1: theta_phys = np.atleast_2d(theta_phys)

    p = x_phys.shape[0] # Number of x points
    n = theta_phys.shape[0] # Number of theta points

    # Stack inputs to evaluate all combinations via borehole_vec
    theta_stacked = np.repeat(theta_phys, repeats=p, axis=0)
    x_stacked = np.tile(x_phys, (n, 1))

    # Calculate output and reshape
    f = borehole_vec(x_stacked, theta_stacked).reshape((n, p))

    # Transpose to get shape (num_x_points, num_theta_points)
    # For single x/theta (1,1), returns (1,1)
    return f.T


# --- Wrapper Function for main.py ---

def run_simulator_borehole(combined_inputs_std):
    """
    Runs the Borehole simulator for each row of combined_inputs_std.

    Args:
        combined_inputs_std (np.ndarray): Array of shape (m, d) where d = xdim + thetadim.
                                          Each row contains standardized x and theta values
                                          concatenated, i.e., [xstd_1..xdim, tstd_1..thetadim].
                                          Inputs MUST be in the range [0, 1].
                                          Expected order: [rw, Hl, T_eff, Hu, Ld/Kw, powparam] (based on tstd2theta/xstd2x slicing)

    Returns:
        np.ndarray: Array of shape (m, 1) containing the scalar borehole output
                    for each input row.
    """
    m, total_dim = combined_inputs_std.shape
    xdim = _dict['xdim']
    thetadim = _dict['thetadim']

    if total_dim != (xdim + thetadim):
        raise ValueError(f"Input dimension {total_dim} != xdim+thetadim ({xdim}+{thetadim})")

    outputs = np.zeros((m, 1)) # Borehole output is scalar

    for i in range(m):
        # Extract standardized x and theta for this run
        # Ensure slicing order matches the expected order in xstd2x and tstd2theta
        x_std_i = combined_inputs_std[i:i+1, :xdim]      # First xdim columns are x_std
        theta_std_i = combined_inputs_std[i:i+1, xdim:]  # Remaining columns are theta_std

        # Call the core model function with standardized inputs.
        # It expects potentially multiple inputs, but we pass (1, xdim) and (1, thetadim),
        # so the output shape will be (1, 1).
        output_i = borehole_model(x_std_i, theta_std_i)
        outputs[i, 0] = output_i[0, 0] # Extract the scalar value

    return outputs


# --- Optional: Self-Test Block ---
if __name__ == "__main__":
    print(f"Running self-test for {__name__}")

    # Test the basic model call
    test_x_std = np.array([[0.5, 0.5]]) # Middle of range for rw, Hl
    test_theta_std = np.array([[0.5, 0.5, 0.5, 0.5]]) # Middle of range for T_eff, Hu, Ld/Kw, powparam
    output_single = borehole_model(test_x_std, test_theta_std)
    print(f"\nTest borehole_model(mid, mid):")
    print(f"Output shape: {output_single.shape}")
    print(output_single)

    # Test the run_simulator wrapper
    num_samples = 5
    xdim_test = _dict['xdim']
    thetadim_test = _dict['thetadim']
    combined_inputs_test = np.random.rand(num_samples, xdim_test + thetadim_test)
    print(f"\nTesting run_simulator_borehole with {num_samples} samples...")
    simulator_outputs = run_simulator_borehole(combined_inputs_test)
    print(f"Input shape: {combined_inputs_test.shape}")
    print(f"Output shape: {simulator_outputs.shape}")
    print("Outputs:")
    print(simulator_outputs)

    # Verify dimension consistency
    assert simulator_outputs.shape == (num_samples, 1)
    print("\nSelf-test complete.")