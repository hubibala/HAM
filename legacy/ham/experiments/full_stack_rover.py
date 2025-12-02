import os
import sys

# Force JAX to use CPU only (avoid Metal/StableHLO issues on Mac)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.tree_util import register_dataclass
from dataclasses import dataclass
import optax
import matplotlib.pyplot as plt
import numpy as np

# --- IMPORTS FROM OUR STACK ---
from ham.solvers import AVBDSolver
from ham.geometry import parallel_transport

# Assuming spherical_rover has these components.
# If not, they are redefined below for self-containment.
from ham.experiments.spherical_rover import (
    init_spherical_net,
    metric_fn,
    spherical_constraint,
    plot_spherical_world,
)

# --- HELPER: FINSLER ENERGY (The Physics Engine) ---


@register_dataclass
@dataclass(frozen=True)
class RandersMetric:
    L: jnp.ndarray
    beta: jnp.ndarray


def metric_adapter(theta, p):
    """Adapts raw MLP output to RandersMetric struct."""
    g, beta = metric_fn(theta, p)
    # g is Identity for this experiment, so L is Identity
    return RandersMetric(L=jnp.eye(3), beta=beta)


def discrete_randers_energy(path, metric_fn_bound):
    """
    Computes the action of the path under the learned physics.
    This is what allows the Navigator to 'see' the wind.
    """
    velocities = path[1:] - path[:-1]
    midpoints = (path[1:] + path[:-1]) / 2.0

    def step_energy(v, x):
        m = metric_fn_bound(x)
        Lv = jnp.dot(m.L.T, v)
        alpha = jnp.sqrt(jnp.dot(Lv, Lv) + 1e-9)
        drift = jnp.dot(m.beta, v)
        return 0.5 * (alpha + drift) ** 2

    energies = vmap(step_energy)(velocities, midpoints)
    return jnp.sum(energies)


# --- 1. THE PILOT (RL / Controller) ---


class PilotPolicy:
    """
    The 'Muscle Memory' of the agent.
    It lives in the Tangent Space (2D local frame).
    """

    def __init__(self, key):
        self.params = {
            "w1": random.normal(key, (2, 32)) * 0.1,
            "b1": jnp.zeros(32),
            "w2": random.normal(key, (32, 4)) * 0.1,
            "b2": jnp.zeros(4),
        }

    @staticmethod
    def forward(params, v_cmd):
        h = jax.nn.relu(jnp.dot(v_cmd, params["w1"]) + params["b1"])
        u = jax.nn.softplus(jnp.dot(h, params["w2"]) + params["b2"])
        return u


# --- 2. THE LAB BENCH (Flat Training Environment) ---


def train_pilot_on_bench():
    print("\n--- Step 1: Pre-training Pilot on Flat Bench ---")
    key = random.PRNGKey(101)
    pilot = PilotPolicy(key)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(pilot.params)

    thruster_dirs = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    @jit
    def loss_fn(params, v_cmd_batch):
        def single_loss(v_cmd):
            u = PilotPolicy.forward(params, v_cmd)
            v_actual = jnp.dot(u, thruster_dirs)
            return jnp.sum((v_actual - v_cmd) ** 2)

        return jnp.mean(vmap(single_loss)(v_cmd_batch))

    @jit
    def step(params, opt_state, key):
        v_cmds = random.normal(key, (32, 2))
        loss, grads = jax.value_and_grad(loss_fn)(params, v_cmds)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(200):
        key, subkey = random.split(key)
        pilot.params, opt_state, loss = step(pilot.params, opt_state, subkey)
        if i % 50 == 0:
            print(f"Bench Epoch {i}: Control Error = {loss:.6f}")

    print("Pilot is certified ready for duty.")
    return pilot


# --- 3. THE MISSION (Integration) ---


def run_full_stack_mission():
    # A. Get Pilot
    pilot = train_pilot_on_bench()

    # B. Get Navigator (World Model)
    print("\n--- Step 2: Initializing Navigator (World Model) ---")
    key = random.PRNGKey(2025)

    # Note: In a real pipeline, we would load 'theta_ham' from a trained file.
    # Here, we initialize it. To see the 'Curved Path' effect, we ideally need
    # the "Hurricane" weights. Let's use the Hurricane Init if available,
    # or standard init with high variance to ensure some wind exists.
    theta_ham = init_spherical_net(key)

    # C. Execute Mission on Sphere
    print("\n--- Step 3: Deploying Rover on Sphere ---")
    p_start = jnp.array([1.0, 0.0, 0.0])
    p_end = jnp.array([0.0, 0.0, 1.0])

    # 1. NAVIGATOR: Plans the Geodesic Path
    print("Navigator: Planning optimal Finslerian path...")
    avbd = AVBDSolver(lr=0.05, beta=20.0, max_iters=150)

    def solver_wrapper(th, p1, p2):
        num_inner = 20
        t = jnp.linspace(0, 1, num_inner + 2)[1:-1]
        linear = p1[None, :] * (1 - t[:, None]) + p2[None, :] * t[:, None]
        init_inner = linear / jnp.linalg.norm(linear, axis=1, keepdims=True)

        # CRITICAL FIX: Use Finsler Energy, not Euclidean
        def e_fn(inner):
            full = jnp.concatenate([p1[None, :], inner, p2[None, :]])
            # Bind the learned physics (th) to the energy function
            bound_metric = lambda x: metric_adapter(th, x)
            return discrete_randers_energy(full, bound_metric)

        constraints = [spherical_constraint]
        return avbd.solve(e_fn, constraints, p1, p2, init_inner)

    # Plan path using Learned Physics
    path = solver_wrapper(theta_ham, p_start, p_end)

    # 2. EXECUTION LOOP
    print("Pilot: Engaging Thrusters along path...")
    actual_trajectory = [p_start]
    current_pos = p_start
    dt = 0.1

    def get_local_basis(p):
        k = jnp.array([0.0, 0.0, 1.0])
        east = jnp.cross(k, p)
        norm = jnp.linalg.norm(east)
        east = jax.lax.cond(norm < 1e-3, lambda: jnp.array([0.0, 1.0, 0.0]), lambda: east / norm)
        north = jnp.cross(p, east)
        return jnp.stack([east, north])

    for i in range(len(path) - 1):
        target_pos = path[i + 1]

        # A. Navigator Command (Global 3D)
        v_global = (target_pos - current_pos) / dt

        # B. Transform to Pilot Frame (Local 2D)
        basis = get_local_basis(current_pos)
        v_local_2d = jnp.dot(basis, v_global)

        # C. Pilot Actuation
        thrusters = PilotPolicy.forward(pilot.params, v_local_2d)

        # D. Simulation Physics
        thruster_vecs = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        force_local_2d = jnp.dot(thrusters, thruster_vecs)
        force_global = jnp.dot(force_local_2d, basis)

        # Euler Integration
        next_pos = current_pos + force_global * dt
        next_pos = next_pos / jnp.linalg.norm(next_pos)  # Constraint

        current_pos = next_pos
        actual_trajectory.append(current_pos)

    # --- 4. VISUALIZATION ---
    actual_trajectory = jnp.array(actual_trajectory)

    print("Mission Complete. Visualizing...")
    plot_spherical_world(theta_ham, path, title="Planned (Crimson) vs Actual (Cyan)")

    fig = plt.gcf()
    ax = fig.get_axes()[0]
    path_np = np.array(actual_trajectory)
    ax.plot(
        path_np[:, 0],
        path_np[:, 1],
        path_np[:, 2],
        color="cyan",
        linestyle="--",
        linewidth=2,
        label="Pilot Execution",
    )
    ax.legend()
    plt.savefig("full_stack_rover.png", dpi=150)
    print("Saved to 'full_stack_rover.png'")


if __name__ == "__main__":
    run_full_stack_mission()
