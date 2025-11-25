import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.tree_util import register_dataclass
from dataclasses import dataclass
from typing import Callable, List
import optax

# --- IMPORT YOUR MODULES ---
from ham.solvers import AVBDSolver
from ham.geometry import parallel_transport
from ham.losses import holonomy_error_loss

# Assuming initialization logic exists in experiment utils
from ham.experiments.teleportation_experiment import init_metric_net, metric_fn as mlp_metric_fn

# --- 1. ADAPTERS & DATA STRUCTURES ---


@register_dataclass
@dataclass(frozen=True)
class RandersMetric:
    """Struct to satisfy your energy function's interface."""

    L: jnp.ndarray  # Cholesky of g (or sqrt(g) if diagonal)
    beta: jnp.ndarray  # Wind vector


def discrete_randers_energy(path: jnp.ndarray, metric_fn: Callable) -> jnp.float32:
    """
    YOUR IMPLEMENTATION (Midpoint Rule)
    """
    # 1. Compute Velocities and Midpoints
    velocities = path[1:] - path[:-1]
    midpoints = (path[1:] + path[:-1]) / 2.0

    # 2. Define Step Energy
    def step_energy(v, x):
        m = metric_fn(x)
        # v.T a v = || L.T v ||^2
        Lv = jnp.dot(m.L.T, v)
        # Riemannian part (alpha)
        alpha = jnp.sqrt(jnp.dot(Lv, Lv) + 1e-9)
        # Drift part (beta)
        drift = jnp.dot(m.beta, v)
        # Randers Squared Norm
        return 0.5 * (alpha + drift) ** 2

    # 3. Integrate (Sum)
    energies = vmap(step_energy)(velocities, midpoints)
    return jnp.sum(energies)


def metric_adapter(theta, p):
    """
    Adapts the MLP output (g, beta) to the RandersMetric object.
    """
    g_matrix, beta_vector = mlp_metric_fn(theta, p)

    # Since g is Identity in this experiment, L is Identity.
    # In full HAM, L would be the Cholesky factor of g.
    L = jnp.eye(p.shape[0])

    return RandersMetric(L=L, beta=beta_vector)


# --- 2. EXPERIMENT SETUP ---


def run_avbd_teleportation():
    print("--- Phase D: AVBD Finsler Teleportation (Midpoint Integration) ---")

    # Contexts (90 degree rotation)
    p_A = jnp.array([0.0, 0.0])
    v_skill_A = jnp.array([1.0, 0.0])  # Push East
    p_B = jnp.array([2.0, 2.0])
    v_skill_B_true = jnp.array([0.0, 1.0])  # Push North

    # Init Network & Solver
    key = random.PRNGKey(42)
    theta = init_metric_net(key)

    # Using your solver with moderate learning rate for stability
    avbd = AVBDSolver(lr=0.05, beta=10.0, max_iters=100)

    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(theta)

    # --- 3. SOLVER WRAPPER (The Glue) ---

    def solver_wrapper_avbd(th, p_start, p_end):
        """
        Binds the Metric parameters 'th' to the energy function
        and calls the AVBD solver.
        """
        # Linear Initialization
        num_inner = 18
        t = jnp.linspace(0, 1, num_inner + 2)[1:-1]
        init_inner = p_start[None, :] * (1 - t[:, None]) + p_end[None, :] * t[:, None]

        # Define Energy bound to current theta
        def energy_fn(inner_path):
            # Reconstruct full path here because your energy function expects it
            full_path_candidate = jnp.concatenate(
                [p_start[None, :], inner_path, p_end[None, :]], axis=0
            )

            # Bind theta to the adapter so it looks like f(x) -> RandersMetric
            bound_metric = lambda x: metric_adapter(th, x)

            return discrete_randers_energy(full_path_candidate, bound_metric)

        # Solve
        return avbd.solve(energy_fn, [], p_start, p_end, init_inner)

    # --- 4. TRAINING LOOP ---

    @jit
    def step(theta, opt_state):
        # We differentiate through the AVBD solver!
        loss_val, grads = jax.value_and_grad(holonomy_error_loss)(
            theta,
            p_A,
            v_skill_A,
            p_B,
            v_skill_B_true,
            mlp_metric_fn,
            solver_wrapper_avbd,
            parallel_transport,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss_val

    print("Training Geometry...")
    loss_history = []
    for i in range(100):
        theta, opt_state, loss = step(theta, opt_state)
        loss_history.append(loss)
        if i % 10 == 0:
            print(f"Epoch {i}: Loss = {loss:.6f}")

    # --- 5. VERIFICATION ---
    print("\n--- Zero-Shot Verification ---")

    # A. Get the Curved Path
    final_path = solver_wrapper_avbd(theta, p_A, p_B)

    # B. Transport Skill along Curved Path
    v_new_A = jnp.array([1.0, 1.0])  # "North-East"
    v_new_B_expected = jnp.array([-1.0, 1.0])  # "North-West" (Rotated 90 deg)

    v_transported = parallel_transport(theta, final_path, v_new_A, mlp_metric_fn)

    print(f"Test Skill (A): {v_new_A}")
    print(f"Expected (B):   {v_new_B_expected}")
    print(f"Transported:    {v_transported}")

    cosine_sim = jnp.dot(v_transported, v_new_B_expected) / (
        jnp.linalg.norm(v_transported) * jnp.linalg.norm(v_new_B_expected)
    )
    print(f"Cosine Similarity: {cosine_sim:.4f}")

    if cosine_sim > 0.9:
        print("SUCCESS: Holonomy learnt via AVBD geodesics!")


if __name__ == "__main__":
    run_avbd_teleportation()
