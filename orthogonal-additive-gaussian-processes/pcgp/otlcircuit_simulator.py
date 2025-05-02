# otlcircuit_simulator.py
"""OTL circuit function simulator."""
import numpy as np

_dict = {
    'function': 'OTLcircuit',
    'xdim': 4,
    'thetadim': 2
    # Add physical bounds or input names here if needed for more detailed analysis/plotting later
    # 'xbounds': np.array([[50, 150], [25, 75], [1.2, 2.5], [0.25, 1.2]]),
    # 'thetabounds': np.array([[0.5, 3], [50, 300]]),
    # 'input_names_x': ['Rb1', 'Rb2', 'Rc1', 'Rc2'],
    # 'input_names_theta': ['Rf', 'beta']
}

# --- Helper functions for standardization ---

def tstd2theta(tstd):
    """Given standardized theta in [0, 1]^d, return non-standardized theta."""
    # Defines physical ranges for theta: Rf, beta
    if tstd.ndim < 1.5:
        tstd = np.atleast_2d(tstd)
    Rfs = tstd[:, 0:1]   # Slice to keep 2D
    betas = tstd[:, 1:2] # Slice to keep 2D

    Rf = 0.5 + Rfs * (3 - 0.5)          # Range [0.5, 3] kOhm
    beta = 50 + betas * (300 - 50)      # Range [50, 300]

    theta_phys = np.hstack((Rf, beta))
    return theta_phys

def xstd2x(xstd):
    """Given standardized x in [0, 1]^d, return non-standardized x."""
    # Defines physical ranges for x: Rb1, Rb2, Rc1, Rc2
    if xstd.ndim < 1.5:
        xstd = np.atleast_2d(xstd)
    Rb1s = xstd[:, 0:1]  # Slice to keep 2D
    Rb2s = xstd[:, 1:2]
    Rc1s = xstd[:, 2:3]
    Rc2s = xstd[:, 3:4]

    Rb1 = 50 + Rb1s * (150 - 50)        # Range [50, 150] kOhm
    Rb2 = 25 + Rb2s * (75 - 25)          # Range [25, 75] kOhm
    Rc1 = 1.2 + Rc1s * (2.5 - 1.2)       # Range [1.2, 2.5] kOhm
    Rc2 = 0.25 + Rc2s * (1.2 - 0.25)     # Range [0.25, 1.2] kOhm

    x_phys = np.hstack((Rb1, Rb2, Rc1, Rc2))
    return x_phys

# --- Core Simulation Logic ---

def OTLcircuit_vec(x_phys, theta_phys):
    """
    Calculates midpoint voltage Vm given PHYSICAL inputs x and theta.
    """
    # Split physical inputs
    Rb1 = x_phys[:, 0:1]
    Rb2 = x_phys[:, 1:2]
    Rc1 = x_phys[:, 2:3]
    Rc2 = x_phys[:, 3:4]
    Rf = theta_phys[:, 0:1]
    beta = theta_phys[:, 1:2]

    const1 = 0.74 # Assumed constant voltage drop

    # Calculate intermediate terms (ensuring broadcasting works with potential 2D inputs)
    Vb1 = 12 * Rb2 / (Rb1 + Rb2)
    term1a = (Vb1 + const1) * beta * (Rc2 + 9) # Assuming 9 is Rc load in kOhms?
    term1b = beta * (Rc2 + 9) + Rf
    term1 = term1a / term1b

    term2a = 11.35 * Rf # Assuming 11.35 is related to Vcc - Vb1?
    term2b = beta * (Rc2 + 9) + Rf
    term2 = term2a / term2b

    term3a = const1 * Rf * beta * (Rc2 + 9)
    term3b = (beta * (Rc2 + 9) + Rf) * Rc1
    term3 = term3a / term3b

    Vm = (term1 + term2 + term3).reshape(-1) # Reshape to 1D vector

    return Vm

def OTLcircuit_model(x_std, theta_std):
    """
    Given standardized x_std and standardized theta_std, calculates the OTL output.
    Handles potentially multiple x and theta points.
    """
    theta_phys = tstd2theta(theta_std)
    x_phys = xstd2x(x_std)

    if x_phys.ndim == 1: x_phys = np.atleast_2d(x_phys)
    if theta_phys.ndim == 1: theta_phys = np.atleast_2d(theta_phys)

    p = x_phys.shape[0] # num x points
    n = theta_phys.shape[0] # num theta points

    # Stack for vectorized calculation across all combinations
    theta_stacked = np.repeat(theta_phys, repeats=p, axis=0)
    x_stacked = np.tile(x_phys, (n, 1))

    f = OTLcircuit_vec(x_stacked, theta_stacked).reshape((n, p))

    # Return shape (num_x_points, num_theta_points)
    return f.T

# --- Wrapper Function for main.py ---

def run_simulator_otlcircuit(combined_inputs_std):
    """
    Runs the OTL Circuit simulator for each row of combined_inputs_std.

    Args:
        combined_inputs_std (np.ndarray): Array of shape (m, d) where d = xdim + thetadim.
                                          Each row contains standardized x and theta values
                                          concatenated. Assumed order: [x_std..., theta_std...].
                                          Inputs MUST be in the range [0, 1].

    Returns:
        np.ndarray: Array of shape (m, 1) containing the scalar Vm output
                    for each input row.
    """
    m, total_dim = combined_inputs_std.shape
    xdim = _dict['xdim']
    thetadim = _dict['thetadim']

    if total_dim != (xdim + thetadim):
        raise ValueError(f"Input dimension {total_dim} != xdim+thetadim ({xdim}+{thetadim})")

    outputs = np.zeros((m, 1)) # OTL Circuit output is scalar

    for i in range(m):
        # Extract standardized x and theta for this run
        x_std_i = combined_inputs_std[i:i+1, :xdim]      # Shape (1, xdim)
        theta_std_i = combined_inputs_std[i:i+1, xdim:]  # Shape (1, thetadim)

        # Call the core model function. Output shape will be (1, 1).
        output_i = OTLcircuit_model(x_std_i, theta_std_i)
        outputs[i, 0] = output_i[0, 0] # Extract the scalar value

    return outputs


# --- Optional: Self-Test Block ---
if __name__ == "__main__":
    print(f"Running self-test for {__name__}")

    # Test the basic model call
    test_x_std = np.array([[0.5] * _dict['xdim']]) # Mid-range standardized x
    test_theta_std = np.array([[0.5] * _dict['thetadim']]) # Mid-range standardized theta
    output_single = OTLcircuit_model(test_x_std, test_theta_std)
    print(f"\nTest OTLcircuit_model(mid_x, mid_theta):")
    print(f"Output shape: {output_single.shape}")
    print(output_single)

    # Test the run_simulator wrapper
    num_samples = 5
    xdim_test = _dict['xdim']
    thetadim_test = _dict['thetadim']
    combined_inputs_test = np.random.rand(num_samples, xdim_test + thetadim_test)
    print(f"\nTesting run_simulator_otlcircuit with {num_samples} samples...")
    simulator_outputs = run_simulator_otlcircuit(combined_inputs_test)
    print(f"Input shape: {combined_inputs_test.shape}")
    print(f"Output shape: {simulator_outputs.shape}")
    print("Outputs:")
    print(simulator_outputs)

    # Verify dimension consistency
    assert simulator_outputs.shape == (num_samples, 1)
    print("\nSelf-test complete.")