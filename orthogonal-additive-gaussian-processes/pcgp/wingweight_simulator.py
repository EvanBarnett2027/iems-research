import numpy as np

_dict = {
    'function': 'Wingweight',
    'xdim': 6,
    'thetadim': 4,
    'xbounds': np.array([[150, 200],
                         [220, 300],
                         [0.08, 0.18],
                         [2.5, 6],
                         [1700, 2500],
                         [0.025, 0.08]]),
    'thetabounds': np.array([[6, 10],
                             [-10, 10],
                             [16, 45],
                             [0.5, 1]]),
    # fail_array and c_array can be included if Wingweight_failmodel is used
    # For now, they are commented out as they are not used by Wingweight_model directly
    # 'c_array': np.arange(0.5, 1.5, 0.0005),
    # 'fail_array': np.array([...]) # Your long array
}

def xstd2x(xstd):
    # Takes STANDARDIZED xstd [0,1]
    if xstd.ndim < 1.5:
        xstd = np.atleast_2d(xstd)
    bounds = _dict['xbounds']
    x = bounds[:, 0] + xstd * (bounds[:, 1] - bounds[:, 0])
    return x

def tstd2theta(tstd):
    # Takes STANDARDIZED tstd [0,1]
    if tstd.ndim < 1.5:
       tstd = np.atleast_2d(tstd)
    bounds = _dict['thetabounds']
    theta = bounds[:, 0] + tstd * (bounds[:, 1] - bounds[:, 0])
    return theta

def Wingweight_vec(x_phys, theta_phys):
    # Takes PHYSICAL inputs
    (Sw, Wfw, tc, Nz, Wdg, Wp) = np.split(x_phys, x_phys.shape[1], axis=1)
    (A, LamCaps, q, lam) = np.split(theta_phys, theta_phys.shape[1], axis=1)

    LamCaps = LamCaps * (np.pi / 180.) # Convert degrees to radians

    fact1 = 0.036 * Sw ** 0.758 * Wfw ** 0.0035
    fact2 = (A / (np.cos(LamCaps) ** 2)) ** 0.6
    fact3 = q ** 0.006 * lam ** 0.04
    fact4 = (100 * tc / np.cos(LamCaps)) ** (-0.3)
    fact5 = (Nz * Wdg) ** 0.49
    term1 = Sw * Wp

    W = (fact1 * fact2 * fact3 * fact4 * fact5 + term1).reshape(-1)
    return W

def Wingweight_model(x_std, theta_std):
    """
    Given standardized x_std and standardized theta_std,
    return matrix of [row x_std] times [row theta_std] of values.
    """
    # Important: This function expects STANDARDIZED inputs x and theta in [0, 1]
    theta_phys = tstd2theta(theta_std) # Convert theta to physical scale
    x_phys = xstd2x(x_std)           # Convert x to physical scale

    # p is number of x points, n is number of theta points
    # Wingweight_model is typically called with single x and single theta for our GP data generation
    # but we'll keep its original structure for compatibility if it expects multiple.
    if x_phys.ndim == 1: x_phys = np.atleast_2d(x_phys)
    if theta_phys.ndim == 1: theta_phys = np.atleast_2d(theta_phys)

    p = x_phys.shape[0]
    n = theta_phys.shape[0]

    # Stack inputs for vectorized calculation in Wingweight_vec
    # This creates all combinations of x_phys rows with theta_phys rows
    theta_stacked = np.repeat(theta_phys, repeats=p, axis=0)
    x_stacked = np.tile(x_phys, (n, 1)) # tile x_phys 'n' times

    # Calculate output and reshape
    f = Wingweight_vec(x_stacked, theta_stacked).reshape((n, p))
    # Return shape is (num_theta_points, num_x_points)
    # For consistency with typical (num_samples, output_dim), we transpose if p=1 or n=1
    # If called with single x and single theta (1,1), it becomes (1,1) after transpose.
    # If called with multiple x and single theta (p,1), it becomes (p,1) after transpose.
    return f.T


def Wingweight_true(x_std):
    """Given standardized x_std, return matrix of [row x] times 1 of values."""
    # assume true theta is [0.5]^d (standardized)
    theta0_std = np.atleast_2d(np.array([0.5] * _dict['thetadim']))
    f0 = Wingweight_model(x_std, theta0_std) # Uses standardized theta internally
    return f0

def run_simulator_wingweight(combined_inputs_std):
    """
    Runs the Wingweight simulator for each row of combined_inputs_std.

    Args:
        combined_inputs_std (np.ndarray): Array of shape (m, d) where d = xdim + thetadim.
                                          Each row contains the standardized x and theta values
                                          concatenated, i.e., [xstd_1..xstd_6, tstd_1..tstd_4].
                                          Inputs MUST be in the range [0, 1].

    Returns:
        np.ndarray: Array of shape (m, 1) containing the scalar wing weight output
                    for each input row.
    """
    m, total_dim = combined_inputs_std.shape
    xdim = _dict['xdim']
    thetadim = _dict['thetadim']

    if total_dim != (xdim + thetadim):
        raise ValueError(f"Input dimension {total_dim} does not match xdim+thetadim ({xdim}+{thetadim})")

    outputs = np.zeros((m, 1)) # Wingweight output is scalar

    for i in range(m):
        # Extract standardized x and theta for this run
        x_std_i = combined_inputs_std[i:i+1, :xdim]    # Shape (1, xdim)
        theta_std_i = combined_inputs_std[i:i+1, xdim:] # Shape (1, thetadim)

        # Call the core model function.
        # It returns shape (1, 1) when called with single x_std_i and theta_std_i.
        output_i = Wingweight_model(x_std_i, theta_std_i)
        outputs[i, 0] = output_i[0, 0]

    return outputs

# Example of how to test this file independently (optional)
if __name__ == "__main__":
    print("Testing wingweight_simulator.py...")
    x_test_std = np.random.rand(2, _dict['xdim'])
    theta_test_std = np.random.rand(3, _dict['thetadim'])

    # Test Wingweight_model directly
    print("\nTesting Wingweight_model with multiple x and multiple theta:")
    # Will produce a (2, 3) output matrix (num_x, num_theta)
    output_matrix = Wingweight_model(x_test_std, theta_test_std)
    print(f"Output shape: {output_matrix.shape}")
    print(output_matrix)

    print("\nTesting Wingweight_model with single x and single theta:")
    output_scalar = Wingweight_model(x_test_std[0:1,:], theta_test_std[0:1,:])
    print(f"Output shape: {output_scalar.shape}")
    print(output_scalar)


    # Test run_simulator_wingweight
    print("\nTesting run_simulator_wingweight:")
    num_samples = 5
    combined_inputs_test = np.random.rand(num_samples, _dict['xdim'] + _dict['thetadim'])
    simulator_outputs = run_simulator_wingweight(combined_inputs_test)
    print(f"Combined inputs shape: {combined_inputs_test.shape}")
    print(f"Simulator outputs shape: {simulator_outputs.shape}") # Should be (num_samples, 1)
    print(simulator_outputs)

    print("\nTesting Wingweight_true:")
    true_vals = Wingweight_true(x_test_std)
    print(f"True values shape: {true_vals.shape}") # Should be (num_x_test, 1)
    print(true_vals)