import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax  # Standard JAX optimizer library
import matplotlib.pyplot as plt

# Import our verified engine
from ham.solvers import AVBDSolver
from ham.geometry import parallel_transport
from ham.losses import holonomy_error_loss

# --- 1. THE WORLD MODEL (Metric Network) ---


def init_metric_net(key, input_dim=2, hidden_dim=64):
    """Simple MLP that outputs the Wind field (beta) at any point p."""
    # We output 2 values for beta (2D wind).
    # The Riemannian part 'g' is kept Identity for this specific test
    # to isolate the "Wind-driven Holonomy" effect.
    w1 = random.normal(key, (input_dim, hidden_dim)) * 0.1
    b1 = jnp.zeros(hidden_dim)
    w2 = random.normal(key, (hidden_dim, input_dim)) * 0.1
    b2 = jnp.zeros(input_dim)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def metric_fn(theta, p):
    """Evaluates the metric at point p."""
    # 1. Forward Pass (MLP)
    h = jnp.tanh(jnp.dot(p, theta["w1"]) + theta["b1"])
    beta = jnp.dot(h, theta["w2"]) + theta["b2"]

    # 2. Constraints (Randers Stability)
    # Squash beta so |beta| < 1 (Zermelo condition)
    beta = 0.95 * jnp.tanh(beta)

    # 3. Return (g, beta)
    # g is Identity (Euclidean base space)
    g = jnp.eye(p.shape[0])
    return g, beta


# --- 2. THE EXPERIMENT ---


def run_teleportation_experiment():
    print("--- Phase D: The Teleportation Experiment ---")

    # A. Setup Contexts
    # Context A: Located at (0,0). "East" is relevant here.
    p_A = jnp.array([0.0, 0.0])
    v_skill_A = jnp.array([1.0, 0.0])  # "Push East"

    # Context B: Located at (2,2). "North" is relevant here.
    # We imagine the world has rotated 90 degrees between A and B.
    p_B = jnp.array([2.0, 2.0])
    v_skill_B_true = jnp.array([0.0, 1.0])  # "Push North"

    # B. Initialization
    key = random.PRNGKey(42)
    theta = init_metric_net(key)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(theta)

    # Wrappers for our engine to fit the loss function signature
    # (Solver assumes linear init for this toy test to speed up training,
    #  but in full HAM, this would call avbd_solver.solve)
    def solver_wrapper(th, p1, p2):
        return jnp.linspace(p1, p2, 20)  # Simple linear path for now

    # C. Training Loop (Meta-Learning the Geometry)
    print(f"Goal: Learn that moving A->B requires 90-degree rotation.")

    @jit
    def step(theta, opt_state):
        # Calculate loss and gradients
        loss_val, grads = jax.value_and_grad(holonomy_error_loss)(
            theta,
            p_A,
            v_skill_A,
            p_B,
            v_skill_B_true,
            metric_fn,
            solver_wrapper,
            parallel_transport,
        )
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss_val

    loss_history = []
    for i in range(200):
        theta, opt_state, loss = step(theta, opt_state)
        loss_history.append(loss)
        if i % 20 == 0:
            print(f"Epoch {i}: Loss = {loss:.6f}")

    # D. Validation: Zero-Shot Transfer
    print("\n--- Testing Zero-Shot Transfer ---")

    # 1. Transport the original skill
    path = solver_wrapper(theta, p_A, p_B)
    v_transported = parallel_transport(theta, path, v_skill_A, metric_fn)
    print(f"Original Skill (A): {v_skill_A}")
    print(f"Target Skill (B):   {v_skill_B_true}")
    print(f"Transported:        {v_transported}")

    # 2. Transport a NEW skill (Generalization)
    # If the geometry is truly learned, it should rotate ANY vector 90 degrees.
    # Let's try "Push North-East" (1, 1) -> Should become "North-West" (-1, 1)
    v_new_A = jnp.array([1.0, 1.0])
    v_new_B_expected = jnp.array([-1.0, 1.0])  # Rotated 90 deg

    v_new_transported = parallel_transport(theta, path, v_new_A, metric_fn)

    print(f"\n[Generalization Check]")
    print(f"New Skill (A):      {v_new_A}")
    print(f"Expected (B):       {v_new_B_expected}")
    print(f"Transported:        {v_new_transported}")

    # Check alignment
    cosine_sim = jnp.dot(v_new_transported, v_new_B_expected) / (
        jnp.linalg.norm(v_new_transported) * jnp.linalg.norm(v_new_B_expected)
    )
    print(f"Cosine Similarity:  {cosine_sim:.4f}")

    if cosine_sim > 0.9:
        print("SUCCESS: The World Model learned the hidden symmetry!")
    else:
        print("FAILURE: Model overfitted or failed to capture rotation.")


if __name__ == "__main__":
    run_teleportation_experiment()
