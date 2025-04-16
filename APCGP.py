import itertools
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.kernels import Kernel, Sum, Product, RBF, Constant
from gpflow.likelihoods import Gaussian
from gpflow.models import GPR
from gpflow.config import set_default_float, default_float
from gpflow.utilities import positive, print_summary
from gpflow.kernels.multioutput import LinearCoregionalization

set_default_float(tf.float64)

# -----------------------------------------------------------------------------
# Section 1: Custom Orthogonal Kernel (NEEDS IMPLEMENTATION)
# -----------------------------------------------------------------------------
# This is the most crucial part - implementing the orthogonal constraint
# from Lu et al. (2022). This is a placeholder.
# You'll need to decide on the base kernel (e.g., RBF) and how to handle
# the input distribution p_i(x_i) for orthogonalization (e.g., assume
# Gaussian, use empirical measure).

class OrthogonalKernelBase(Kernel):
    """
    Base class for kernels satisfying orthogonality constraints.
    Requires implementation based on Lu et al. (2022), Eq 8/10 or Appendix E.
    """
    def __init__(self, base_kernel: Kernel, input_dim_info, active_dims=None, name=None):
        super().__init__(active_dims=active_dims, name=name)
        self.base_kernel = base_kernel
        # input_dim_info: Information needed for orthogonalization specific
        # to the dimension this kernel acts on. Could be moments (mean, var)
        # if assuming Gaussian inputs, or empirical data points.
        self.input_dim_info = input_dim_info
        # Ensure base_kernel hyperparameters are tracked
        self.parameters = self.base_kernel.parameters


    def K(self, X, X2=None):
        # --- !!! IMPLEMENTATION REQUIRED !!! ---
        # Calculate the base kernel matrix K_base = self.base_kernel(X, X2)
        # Calculate the expectation terms based on self.input_dim_info
        # (e.g., E[k(x, a)], E[k(a, b)]) - This might involve integrals or sums.
        # Apply the orthogonalization formula (e.g., Lu et al. Eq 8)
        # K_ortho = K_base - E[k(x, a)] * (E[k(a, b)])^-1 * E[k(b, x')]
        # Return K_ortho
        # Placeholder: Just return base kernel for now
        print("Warning: OrthogonalKernelBase.K is only a placeholder!")
        return self.base_kernel(X, X2)

    def K_diag(self, X):
        # --- !!! IMPLEMENTATION REQUIRED !!! ---
        # Similar to K, but compute only the diagonal K_ortho(x, x)
        # Placeholder: Just return base kernel diag for now
        print("Warning: OrthogonalKernelBase.K_diag is only a placeholder!")
        return self.base_kernel.K_diag(X)

# Example usage with RBF base kernel
# You would need to subclass OrthogonalKernelBase for a specific base kernel
class OrthogonalRBF(OrthogonalKernelBase):
     def __init__(self, variance=1.0, lengthscales=1.0, input_dim_info=None, active_dims=None, name=None):
         # Example: assume RBF base kernel
         base = RBF(variance=variance, lengthscales=lengthscales, active_dims=active_dims)
         super().__init__(base_kernel=base, input_dim_info=input_dim_info, active_dims=active_dims, name=name)
         # Store parameters directly if needed
         self.variance = self.base_kernel.variance
         self.lengthscales = self.base_kernel.lengthscales

# -----------------------------------------------------------------------------
# Section 2: Function to Create Additive Kernel for one g^(j)
# -----------------------------------------------------------------------------

def create_additive_latent_kernel(
    input_dim: int,
    max_interaction_order: int,
    input_dist_info: list, # List of info needed by OrthogonalKernelBase for each dim
    base_kernel_class=OrthogonalRBF, # Use the custom orthogonal kernel
    name_prefix: str = "g"
    ) -> Sum:
    """
    Creates the summed kernel k^{(j)} = sum_u sigma_u^2 * k_u^{(j)}
    for a single latent additive GP g^(j).
    """
    component_kernels = []
    # Optional: Add constant term (bias/offset)
    # bias_variance = tf.Variable(1.0, dtype=default_float(), name=f"{name_prefix}_const_var", trainable=True)
    # gpflow.set_parameter(bias_variance, Parameter(bias_variance, transform=positive())) # Old way - now usually part of Constant kernel
    # component_kernels.append(Constant(variance=bias_variance)) # Using tf.Variable directly here might need Parameter wrapper depending on context

    # Add main effects and interactions up to max_interaction_order
    for order in range(1, max_interaction_order + 1):
        for u_indices in itertools.combinations(range(input_dim), order):
            # u_indices is a tuple like (0,) or (0, 2) or (1, 3, 4)

            # Create the product kernel for this interaction term u
            prod_components = []
            for i in u_indices:
                # Create the orthogonal 1D kernel for dimension i
                ortho_kernel_i = base_kernel_class(
                    active_dims=[i],
                    input_dim_info=input_dist_info[i]
                    # The base kernel (e.g., OrthogonalRBF) should internally
                    # handle its trainable parameters like lengthscales using tf.Variable
                )
                prod_components.append(ortho_kernel_i)

            if len(prod_components) == 1:
                term_kernel_u_base = prod_components[0]
            else:
                term_kernel_u_base = Product(prod_components)

            # Create a trainable variance parameter sigma_u^2 for this component kernel
            # Use gpflow.kernels.Constant for the scaling variance
            variance_u_kernel = Constant(
                variance=1.0, # Initial value
                name=f"{name_prefix}_var_u{'_'.join(map(str, u_indices))}"
            )
            # Ensure the variance is positive
            variance_u_kernel.variance.transform = positive()
            # Apply prior if needed (example Gamma prior)
            # variance_u_kernel.variance.prior = tf.compat.v1.distributions.Gamma(1.0, 1.0) # Example prior

            # Scale the base term kernel by its variance: sigma_u^2 * k_u
            # GPflow's * operator handles kernel multiplication
            scaled_term_kernel_u = variance_u_kernel * term_kernel_u_base
            component_kernels.append(scaled_term_kernel_u)

    if not component_kernels:
         raise ValueError("No component kernels were created. Check input_dim and max_interaction_order.")

    # The final kernel for g^(j) is the sum of all scaled component kernels

# -----------------------------------------------------------------------------
# Section 3: Additive PCGP Model Definition
# -----------------------------------------------------------------------------

class AdditivePCGP(GPR):
    """
    Additive PCGP model f = Phi @ g, where each g^(j) is an Additive GP.
    Uses LinearCoregionalization kernel.
    """
    def __init__(self, data, q: int,
                 max_interaction_order: int,
                 input_dist_info: list, # Info needed for OrthogonalKernelBase
                 Phi: tf.Tensor = None, # Optionally provide Phi
                 pca_epsilon: float = 1e-6, # Threshold for PCA component selection
                 pca_center: bool = True, # Center data before PCA?
                 noise_variance: float = 1.0,
                 base_kernel_class=OrthogonalRBF,
                 mean_function=None,
                 phi_trainable: bool = False, # Make Phi trainable?
                 name=None):
        """
        Args:
            data (Tuple[tf.Tensor, tf.Tensor]): X (N, D), Y (N, P) training data.
            
            q (int): Number of latent GPs (desired latent dimension).
            
            max_interaction_order (int): Max order for additive components.
            
            input_dist_info (list): List of info needed for orthogonal kernels.
            
            Phi (tf.Tensor, optional): Pre-computed Basis matrix (P, Q).
                                       If None, computes via PCA on Y. Defaults to None.
            
            pca_epsilon (float, optional): Threshold for variance explained in PCA,
                                           or number of components if q is specified.
                                           Used only if Phi is None. Defaults to 1e-6.
            
            pca_center (bool, optional): Whether to center Y before PCA. Defaults to True.
            
            noise_variance (float, optional): Initial noise variance. Defaults to 1.0.
            
            base_kernel_class (Type[OrthogonalKernelBase], optional): Custom orthogonal kernel.
            
            mean_function (gpflow.mean_functions.MeanFunction, optional): Mean function. Defaults to None (Zero).
            
            phi_trainable (bool, optional): If True, makes the computed/provided Phi trainable.
                                            Defaults to False.
            
            name (str, optional): Model name. Defaults to None.
        """
        X, Y = data
        N, input_dim = X.shape
        N_y, output_dim = Y.shape # P
        assert N == N_y

        # -------------------------------------------------
        # Compute Phi via PCA if not provided
        # -------------------------------------------------
        
        if Phi is None:
            print(f"Phi not provided. Computing {q} principal components from Y...")
            Y_np = Y.numpy()
            if pca_center:
                Y_mean = np.mean(Y_np, axis=0, keepdims=True)
                Y_centered = Y_np - Y_mean
            else:
                Y_centered = Y_np

            # SVD on Y (or Y_centered) directly (N x P)
            # U_svd (N x min(N,P)), S_svd (min(N,P),), Vh_svd (min(N,P) x P)
            # The principal component directions (loadings) are in Vh.T or V
            try:
                 # Need V, which corresponds to right singular vectors (loadings)
                 # np.linalg.svd returns Vh (V transpose)
                 _ , S_svd, Vh_svd = np.linalg.svd(Y_centered, full_matrices=False)
                 V_svd = Vh_svd.T # Shape (P, min(N, P))
            except np.linalg.LinAlgError as e:
                 print(f"SVD failed: {e}. Ensure Y does not contain NaNs/Infs.")
                 raise

            # Select top Q components (principal directions)
            # Note: Your original code selected based on variance threshold,
            #       here we use the specified latent dimension q directly.
            if q > V_svd.shape[1]:
                 raise ValueError(f"Requested latent dimension q={q} is greater than "
                                  f"max possible rank min(N, P)={V_svd.shape[1]}")
            Phi_computed = V_svd[:, :q] # Shape (P, Q)
            Phi = tf.constant(Phi_computed, dtype=default_float())
            print(f"Computed Phi with shape: {Phi.shape}")

        # Ensure provided/computed Phi has correct shape
        tf.debugging.assert_shapes([(Phi, ('P', 'Q'))], message="Phi shape mismatch")
        tf.debugging.assert_equal(tf.shape(Phi)[0], output_dim, message="Phi rows must match output_dim P")
        tf.debugging.assert_equal(tf.shape(Phi)[1], q, message="Phi cols must match latent_dim q")
        # -------------------------------------------------

        # 1. Create the q independent additive latent kernels k^(j)
        latent_kernels = []
        for j in range(q):
            k_j = create_additive_latent_kernel(
                input_dim=input_dim,
                max_interaction_order=max_interaction_order,
                input_dist_info=input_dist_info,
                base_kernel_class=base_kernel_class,
                name_prefix=f"g{j}"
            )
            latent_kernels.append(k_j)

        # 2. Create the Linear Coregionalization Kernel
        # GPflow's W is (Q, P), which corresponds to Phi^T
        # If Phi (and thus W = Phi.T) should be learned/optimized:
        W = tf.Variable(tf.transpose(Phi), trainable=phi_trainable, name="Phi_T", dtype=default_float())
        # Add priors or constraints to W if needed, e.g., for orthogonality if desired # Treat Phi as fixed

        coreg_kernel = LinearCoregionalization(
            kernels=latent_kernels,
            W=W
        )

        # 3. Define Likelihood
        likelihood = Gaussian(variance=noise_variance)

        # 4. Initialize the GPR model
        # GPR calculates the marginal likelihood automatically
        super().__init__(data=data,
                         kernel=coreg_kernel,
                         mean_function=mean_function,
                         noise_variance=noise_variance, # Redundant if using Gaussian likelihood above
                         name=name)

        # Store Phi if needed (W holds Phi^T)
        self.Phi = tf.Variable(Phi, trainable=phi_trainable, name="Phi", dtype=default_float())

    # The log_marginal_likelihood method is inherited from GPR
    # and uses self.kernel, self.likelihood evaluated on self.data

    # Prediction methods (predict_f, predict_y) are also inherited
    # They will use the LinearCoregionalization kernel structure correctly.


# -----------------------------------------------------------------------------
# Section 4: Example Usage
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Dummy Data ---
    N = 50
    D = 3
    P = 5 # More outputs than latent dims now
    Q = 2 # Latent dimension (q < p is typical for PCGP)

    X = tf.random.normal((N, D), dtype=default_float())
    # Generate Y with some structure for PCA to find
    G_true = tf.random.normal((N, Q), dtype=default_float())
    Phi_true = tf.random.normal((P, Q), dtype=default_float())
    F_true = G_true @ tf.transpose(Phi_true)
    Y = F_true + tf.random.normal((N, P), stddev=0.1, dtype=default_float()) # Add noise


    # --- Information for Orthogonal Kernels ---
    input_dist_info = [{'mean': tf.zeros(1, dtype=default_float()),
                        'var': tf.ones(1, dtype=default_float())}] * D

    # --- Model Initialization (Compute Phi internally) ---
    max_order = 2
    model_pca = AdditivePCGP(
        data=(X, Y),
        q=Q,
        Phi=None, # Let the model compute Phi via PCA
        max_interaction_order=max_order,
        input_dist_info=input_dist_info,
        pca_center=True,
        phi_trainable=False # Keep computed Phi fixed initially
    )

    # --- Model Summary and Likelihood ---
    print("--- Model with PCA-derived Phi ---")
    print_summary(model_pca)
    log_likelihood_pca = model_pca.log_marginal_likelihood()
    print(f"\nLog Marginal Likelihood (PCA Phi): {log_likelihood_pca.numpy():.4f}")
    # Access the computed Phi (read-only unless phi_trainable=True)
    # print("Computed Phi:\n", model_pca.Phi.numpy())
    # print("Stored W (Phi.T):\n", model_pca.kernel.W.numpy())

    # --- Example: Provide Phi explicitly ---
    # Phi_fixed = tf.constant(np.random.randn(P,Q), dtype=default_float())
    # model_fixed = AdditivePCGP(
    #     data=(X, Y),
    #     q=Q,
    #     Phi=Phi_fixed, # Provide Phi
    #     max_interaction_order=max_order,
    #     input_dist_info=input_dist_info,
    # )
    # print("\n--- Model with Fixed Phi ---")
    # print_summary(model_fixed)