import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import matplotlib.pyplot as plt

# ==========================================
# 1. 1D Shallow Water Model (SWM) Dynamics
# ==========================================
M = 20         # Number of spatial grid points
DX = 1000.0    # Grid spacing (meters)
DT = 10.0      # Time step (seconds)
G = 9.81       # Gravity acceleration
H0 = 10.0      # Mean fluid depth

NX = 2 * M    
NY = M         # We observe only the height field 'h'

def swm_step(state):
    """Advances the 1D Linearized Shallow Water Model by one time step."""
    h = state[:M]
    u = state[M:]
    
    dh_dx = (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2.0 * DX)
    du_dx = (jnp.roll(h, -1) - jnp.roll(h, 1)) / (2.0 * DX)
    
    h_avg = 0.5 * (jnp.roll(h, -1) + jnp.roll(h, 1))
    u_avg = 0.5 * (jnp.roll(u, -1) + jnp.roll(u, 1))
    
    h_next = h_avg - DT * H0 * dh_dx
    u_next = u_avg - DT * G * du_dx
    
    return jnp.concatenate([h_next, u_next])

# ==========================================
# 2. Spatial Inductive Bias: Periodic 1D CNN
# ==========================================
def conv1d(x, W, b):
    """Periodic 1D Convolution using structural windowing and einsum contraction."""
    k_size, in_ch, out_ch = W.shape
    pad = k_size // 2
    x_pad = jnp.concatenate([x[-pad:], x, x[:pad]], axis=0)
    
    def get_window(i):
        return jax.lax.dynamic_slice(x_pad, (i, 0), (k_size, in_ch))
        
    windows = jax.vmap(get_window)(jnp.arange(x.shape[0]))
    return jnp.einsum('mki,kio->mo', windows, W) + b

def init_cnn_params(key):
    """Initializes a 2-layer periodic 1D CNN for state-dependent adjustments."""
    k1, k2 = random.split(key)
    W1 = random.normal(k1, (5, 2, 8)) * jnp.sqrt(2.0 / (5 * 2))
    b1 = jnp.zeros(8)
    W2 = random.normal(k2, (5, 8, 2)) * jnp.sqrt(2.0 / (5 * 8))
    b2 = jnp.zeros(2)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def cnn_forward(params, x_mean):
    """Processes the mean state to yield a stable modulation vector bounded to [0.8, 1.2]."""
    M_points = x_mean.shape[0] // 2
    x_grid = x_mean.reshape((M_points, 2))  
    
    z1 = jax.nn.relu(conv1d(x_grid, params['W1'], params['b1']))
    z2 = conv1d(z1, params['W2'], params['b2'])
    
    z2_flat = z2.flatten()
    return 0.8 + jax.nn.sigmoid(z2_flat[:, None]) * 0.4

# ==========================================
# 3. Data Assimilation Filters (Updates)
# ==========================================
def mamba_enkf_update(params, X_b, y, H, R, L_env, key):
    """Mamba-style EnKF step enhanced with a physical localization matrix envelope."""
    nx, n_ens = X_b.shape
    ny = y.shape[0]
    
    x_mean = jnp.mean(X_b, axis=1, keepdims=True)
    X_prime = X_b - x_mean
    
    HX = jnp.dot(H, X_b)
    hx_mean = jnp.mean(HX, axis=1, keepdims=True)
    HX_prime = HX - hx_mean
    
    C_xy = jnp.dot(X_prime, HX_prime.T) / (n_ens - 1)
    C_yy = jnp.dot(HX_prime, HX_prime.T) / (n_ens - 1) + R
    
    # Structural Tikhonov regularizer to safely stabilize explicit gain inversions
    C_yy = C_yy + 1e-5 * jnp.eye(ny)
    K_enkf = jnp.dot(C_xy, jnp.linalg.inv(C_yy))
    
    M_mask = cnn_forward(params, x_mean.squeeze())
    K_mod = K_enkf * (L_env * M_mask)
    
    obs_noise = random.multivariate_normal(key, jnp.zeros(ny), R, shape=(n_ens,)).T
    Y_perturbed = y.reshape(ny, 1) + obs_noise
    return X_b + jnp.dot(K_mod, Y_perturbed - HX)

def standard_enkf_update(X_b, y, H, R, key):
    """Standard Stochastic EnKF baseline."""
    nx, n_ens = X_b.shape
    ny = y.shape[0]
    
    x_mean = jnp.mean(X_b, axis=1, keepdims=True)
    X_prime = X_b - x_mean
    
    HX = jnp.dot(H, X_b)
    hx_mean = jnp.mean(HX, axis=1, keepdims=True)
    HX_prime = HX - hx_mean
    
    C_xy = jnp.dot(X_prime, HX_prime.T) / (n_ens - 1)
    C_yy = jnp.dot(HX_prime, HX_prime.T) / (n_ens - 1) + R
    C_yy = C_yy + 1e-5 * jnp.eye(ny)
    K_enkf = jnp.dot(C_xy, jnp.linalg.inv(C_yy))
    
    obs_noise = random.multivariate_normal(key, jnp.zeros(ny), R, shape=(n_ens,)).T
    Y_perturbed = y.reshape(ny, 1) + obs_noise
    return X_b + jnp.dot(K_enkf, Y_perturbed - HX)

def etkf_update(X_b, y, H, R):
    """Standard Deterministic Ensemble Transform Kalman Filter (ETKF)."""
    nx, n_ens = X_b.shape
    ny = y.shape[0]
    
    x_mean = jnp.mean(X_b, axis=1, keepdims=True)
    X_prime = X_b - x_mean
    Y_prime = jnp.dot(H, X_prime)
    R_inv = jnp.linalg.inv(R)
    
    Gamma = (n_ens - 1) * jnp.eye(n_ens) + jnp.dot(Y_prime.T, jnp.dot(R_inv, Y_prime))
    Gamma = Gamma + 1e-5 * jnp.eye(n_ens) # Ridge safety to prevent singular values
    evals, evecs = jnp.linalg.eigh(Gamma)
    Gamma_inv_half = jnp.dot(evecs, jnp.dot(jnp.diag(1.0 / jnp.sqrt(evals)), evecs.T))
    Gamma_inv = jnp.dot(evecs, jnp.dot(jnp.diag(1.0 / evals), evecs.T))
    
    T_mat = jnp.sqrt(n_ens - 1) * Gamma_inv_half
    X_a_prime = jnp.dot(X_prime, T_mat)
    
    innovation = y - jnp.dot(H, x_mean).squeeze()
    d = jnp.dot(Gamma_inv, jnp.dot(Y_prime.T, jnp.dot(R_inv, innovation)))
    x_a_mean = x_mean + jnp.dot(X_prime, d).reshape(nx, 1)
    return x_a_mean + X_a_prime

# ==========================================
# 4. Pure Observation-Space Training Rollout
# ==========================================
def rollout_mamba_enkf_obs_loss(params, X_init, observations, H, R, L_env, keys):
    """REAL-WORLD MODIFICATION: Computes loss using only raw telemetry.
    The true state trajectory has been completely extracted out of the loop.
    """
    R_inv = jnp.linalg.inv(R)

    def step_fn(X_prev, inputs):
        y_t, key_t = inputs
        
        # A. Forecast Step (Prior Model Trajectory)
        X_b = jax.vmap(swm_step, in_axes=1, out_axes=1)(X_prev)
        x_b_mean = jnp.mean(X_b, axis=1, keepdims=True)
        
        # B. Prior Innovation (Forecast vs Instrument sensor reading)
        innovation_prior = y_t - jnp.dot(H, x_b_mean).squeeze()
        
        # C. Filter Assimilation Update Step
        X_a = mamba_enkf_update(params, X_b, y_t, H, R, L_env, key_t)
        x_a_mean = jnp.mean(X_a, axis=1, keepdims=True)
        
        # D. Posterior Innovation (Analysis vs Instrument sensor reading)
        innovation_posterior = y_t - jnp.dot(H, x_a_mean).squeeze()
        
        # E. Multi-Stage Desroziers Loss Configuration
        prior_term = jnp.dot(innovation_prior, jnp.dot(R_inv, innovation_prior))
        posterior_term = jnp.dot(innovation_posterior, jnp.dot(R_inv, innovation_posterior))
        step_loss = 0.5 * prior_term + 0.5 * posterior_term
        
        return X_a, step_loss

    _, step_losses = jax.lax.scan(step_fn, X_init, (observations, keys))
    return jnp.mean(step_losses)

# --- Standard Inference Evaluation Rollouts ---
def rollout_mamba_enkf_inference(params, X_init, observations, H, R, L_env, keys):
    def step_fn(X_prev, inputs):
        y_t, key_t = inputs
        X_b = jax.vmap(swm_step, in_axes=1, out_axes=1)(X_prev)
        X_a = mamba_enkf_update(params, X_b, y_t, H, R, L_env, key_t)
        return X_a, X_a
    _, X_a_history = jax.lax.scan(step_fn, X_init, (observations, keys))
    return X_a_history

def rollout_standard_enkf_inference(X_init, observations, H, R, keys):
    def step_fn(X_prev, inputs):
        y_t, key_t = inputs
        X_b = jax.vmap(swm_step, in_axes=1, out_axes=1)(X_prev)
        X_a = standard_enkf_update(X_b, y_t, H, R, key_t)
        return X_a, X_a
    _, X_a_history = jax.lax.scan(step_fn, X_init, (observations, keys))
    return X_a_history

def rollout_etkf_inference(X_init, observations, H, R):
    def step_fn(X_prev, y_t):
        X_b = jax.vmap(swm_step, in_axes=1, out_axes=1)(X_prev)
        X_a = etkf_update(X_b, y_t, H, R)
        return X_a, X_a
    _, X_a_history = jax.lax.scan(step_fn, X_init, observations)
    return X_a_history

# ==========================================
# 5. Adam Optimizer with Global Norm Clipping
# ==========================================
def adam_update_clipped(step, params, grads, m, v, lr=0.0005, max_norm=0.2):
    flat_grads, _ = jax.tree_util.tree_flatten(grads)
    global_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in flat_grads))
    
    scale = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    clipped_grads = jax.tree.map(lambda g: g * scale, grads)
    
    next_m = jax.tree.map(lambda md, g: 0.9 * md + 0.1 * g, m, clipped_grads)
    next_v = jax.tree.map(lambda vd, g: 0.999 * vd + 0.001 * (g ** 2), v, clipped_grads)
    
    bc1 = 1.0 - 0.9 ** step
    bc2 = 1.0 - 0.999 ** step
    lr_t = lr * jnp.sqrt(bc2) / bc1
    
    next_params = jax.tree.map(
        lambda p, md, vd: p - lr_t * md / (jnp.sqrt(vd) + 1e-8), params, next_m, next_v
    )
    return next_params, next_m, next_v

# ==========================================
# 6. Operational Execution and Validation
# ==========================================
if __name__ == "__main__":
    T, N_ENS, EPOCHS = 50, 30, 10_000
    
    key = random.PRNGKey(88)
    k_init, k_true, k_obs, k_ens, k_train, k_test = random.split(key, 6)
    
    H = jnp.zeros((NY, NX)).at[jnp.arange(M), jnp.arange(M)].set(1.0)
    R = jnp.eye(NY) * 0.04
    
    # Core distance localization envelope arrays
    coord_x = jnp.arange(2 * M) % M
    coord_y = jnp.arange(M)
    dists = jnp.abs(coord_x[:, None] - coord_y[None, :])
    dists = jnp.minimum(dists, M - dists)   
    L_envelope = jnp.exp(-(dists ** 2) / (2.0 * 2.5 ** 2)) 
    
    # --- Data Streaming Setup (Simulating Field Instruments) ---
    x_true_0 = jnp.concatenate([1.0 + 0.6 * jnp.sin(jnp.linspace(0, 2*jnp.pi, M, endpoint=False)), jnp.zeros(M)])
    true_states = []
    curr = x_true_0
    for _ in range(T):
        curr = swm_step(curr)
        true_states.append(curr)
    true_states = jnp.stack(true_states)
    
    # THIS IS OUR ONLY TRAINING INPUT SOURCE
    field_observations = jnp.dot(true_states, H.T) + random.normal(k_obs, (T, NY)) * jnp.sqrt(0.04)
    X_init = x_true_0.reshape(NX, 1) + random.normal(k_ens, (NX, N_ENS)) * 0.4
    
    # --- Optimization Setup ---
    params = init_cnn_params(k_init)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    
    grad_loss_jit = jit(value_and_grad(rollout_mamba_enkf_obs_loss, argnums=0))
    
    print("Beginning Mamba-CNN EnKF Real-World Training Sequence...")
    print("Targeting: Pure Observation-Space Innovation Residuals (No known True States)\n")
    
    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        epoch_keys = random.split(random.fold_in(k_train, epoch), T)
        loss_val, grads = grad_loss_jit(params, X_init, field_observations, H, R, L_envelope, epoch_keys)
        params, m, v = adam_update_clipped(epoch, params, grads, m, v, lr=0.0001, max_norm=0.2)
        loss_history.append(float(loss_val))
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Observation-Space Innovation Loss: {loss_val:.4f}")
            
    # --- Out-of-Sample Verification Data Stream ---
    x_test_true_0 = jnp.concatenate([1.0 + 0.6 * jnp.cos(jnp.linspace(0, 2*jnp.pi, M, endpoint=False)), jnp.zeros(M)])
    test_true_states = []
    curr_test = x_test_true_0
    for _ in range(T):
        curr_test = swm_step(curr_test)
        test_true_states.append(curr_test)
    test_true_states = jnp.stack(test_true_states)
    
    k_t1, k_t2, k_t3 = random.split(k_test, 3)
    test_observations = jnp.dot(test_true_states, H.T) + random.normal(k_t1, (T, NY)) * jnp.sqrt(0.04)
    X_test_init = x_test_true_0.reshape(NX, 1) + random.normal(k_t2, (NX, N_ENS)) * 0.4
    test_scan_keys = random.split(k_t3, T)
    
    # Run Inference
    hist_mamba = rollout_mamba_enkf_inference(params, X_test_init, test_observations, H, R, L_envelope, test_scan_keys)
    hist_std   = rollout_standard_enkf_inference(X_test_init, test_observations, H, R, test_scan_keys)
    hist_etkf  = rollout_etkf_inference(X_test_init, test_observations, H, R)
    
    # Evaluation Verification Metrics (Only used for validation plotting performance post-training)
    mae_mamba = jnp.mean(jnp.abs(jnp.mean(hist_mamba, axis=2) - test_true_states), axis=1)
    mae_std   = jnp.mean(jnp.abs(jnp.mean(hist_std, axis=2) - test_true_states), axis=1)
    mae_etkf  = jnp.mean(jnp.abs(jnp.mean(hist_etkf, axis=2) - test_true_states), axis=1)
    
    # ==========================================
    # 7. 2x2 Diagnostics Plot Block
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    time_axis = jnp.arange(T) * DT
    
    # Top-Left: Pure telemetry-based loss trajectory
    axes[0, 0].plot(range(1, EPOCHS + 1), loss_history, color='teal', lw=2)
    axes[0, 0].set_title("Operational Innovation Loss Trajectory", fontsize=11)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Desroziers Innovation Error (Obs-Space)")
    axes[0, 0].grid(True, linestyle='--')
    
    # Top-Right: Verification Absolute Tracking Errors (Semilogy scale)
    axes[0, 1].semilogy(time_axis, mae_mamba, 'g--', lw=2.5, label='Mamba EnKF')
    axes[0, 1].semilogy(time_axis, mae_std, 'c:', lw=2, label='Standard EnKF')
    axes[0, 1].semilogy(time_axis, mae_etkf, 'm-.', lw=1.5, label='Standard ETKF')
    axes[0, 1].set_title("Validation State Tracking Performance", fontsize=11)
    axes[0, 1].set_xlabel("Simulation Time (Seconds)")
    axes[0, 1].set_ylabel("Mean Absolute Error (Log Scale)")
    axes[0, 1].grid(True, which="both", linestyle='--')
    axes[0, 1].legend()
    
    t_snap = T - 1
    x_axis = jnp.arange(M)
    
    mean_mamba, std_mamba = jnp.mean(hist_mamba[t_snap], axis=1), jnp.std(hist_mamba[t_snap], axis=1)
    mean_std, std_std     = jnp.mean(hist_std[t_snap], axis=1), jnp.std(hist_std[t_snap], axis=1)
    mean_etkf, std_etkf   = jnp.mean(hist_etkf[t_snap], axis=1), jnp.std(hist_etkf[t_snap], axis=1)
    
    # Bottom-Left: Fluid Height Profile Snapshot
    axes[1, 0].plot(x_axis, test_true_states[t_snap, :M], 'k-', lw=2.5, label='Verification Target (Hidden)')
    axes[1, 0].scatter(x_axis, test_observations[t_snap, :], color='red', alpha=0.5, s=25, label='Telemetry Inputs')
    axes[1, 0].plot(x_axis, mean_mamba[:M], 'g--', lw=2, label='Mamba EnKF')
    axes[1, 0].fill_between(x_axis, mean_mamba[:M] - 2*std_mamba[:M], mean_mamba[:M] + 2*std_mamba[:M], color='green', alpha=0.15)
    axes[1, 0].plot(x_axis, mean_std[:M], 'c:', lw=2, label='Standard EnKF')
    axes[1, 0].plot(x_axis, mean_etkf[:M], 'm-.', lw=1.5, label='Standard ETKF')
    axes[1, 0].set_title(f"Fluid Height (h) Profile (t={t_snap*DT}s)", fontsize=11)
    axes[1, 0].set_xlabel("Spatial Coordinates")
    axes[1, 0].grid(True, linestyle='--')
    axes[1, 0].legend()
    
    # Bottom-Right: Unobserved Fluid Velocity Profile Snapshot
    axes[1, 1].plot(x_axis, test_true_states[t_snap, M:], 'k-', lw=2.5, label='Verification Target (Hidden)')
    axes[1, 1].plot(x_axis, mean_mamba[M:], 'g--', lw=2, label='Mamba EnKF')
    axes[1, 1].fill_between(x_axis, mean_mamba[M:] - 2*std_mamba[M:], mean_mamba[M:] + 2*std_mamba[M], color='green', alpha=0.15)
    axes[1, 1].plot(x_axis, mean_std[M:], 'c:', lw=2, label='Standard EnKF')
    axes[1, 1].plot(x_axis, mean_etkf[M:], 'm-.', lw=1.5, label='Standard ETKF')
    axes[1, 1].set_title(f"Unobserved Velocity (u) Profile (t={t_snap*DT}s)", fontsize=11)
    axes[1, 1].set_xlabel("Spatial Coordinates")
    axes[1, 1].grid(True, linestyle='--')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_filename = "mamba_enkf_obs_space_analysis_with_etkf.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nReal-world configuration analysis visualization saved as: '{output_filename}'")
    plt.show()