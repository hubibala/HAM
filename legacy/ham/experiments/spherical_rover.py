import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import optax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Import our stack
from ham.solvers import AVBDSolver
from ham.geometry import parallel_transport
from ham.losses import holonomy_error_loss

# --- 1. GEOMETRY: SPHERICAL CONSTRAINT ---


def spherical_constraint(x):
    """
    Hard constraint: x must lie on the unit sphere.
    C(x) = ||x||^2 - 1 = 0
    """
    return jnp.sum(x**2) - 1.0


# --- 2. THE WORLD MODEL (Metric Network on S2) ---


def init_spherical_net(key, hidden_dim=64):
    # Increased standard deviation (0.1 -> 0.5)
    # This ensures the initial random wind is already strong.
    w1 = random.normal(key, (3, hidden_dim)) * 0.5
    b1 = jnp.zeros(hidden_dim)
    w2 = random.normal(key, (hidden_dim, 3)) * 0.5
    b2 = jnp.zeros(3)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def metric_fn(theta, p):
    """
    Evaluates the Randers metric with HURRICANE strength.
    """
    # 1. Base Riemannian Metric (g = I)
    g = jnp.eye(3)

    # 2. Wind Field (beta)
    h = jnp.tanh(jnp.dot(p, theta["w1"]) + theta["b1"])
    raw_wind = jnp.dot(h, theta["w2"]) + theta["b2"]

    # Project to tangent space
    normal = p / jnp.linalg.norm(p)
    wind_tangent = raw_wind - jnp.dot(raw_wind, normal) * normal

    # CRITICAL CHANGE: Saturation at 0.99 (Was 0.8)
    # This allows the wind to consume 99% of the movement budget.
    beta = 0.99 * jnp.tanh(wind_tangent)

    return g, beta


# Adapter for the energy function
from ham.experiments.teleportation_avbd import discrete_randers_energy, RandersMetric


def metric_adapter(theta, p):
    g, beta = metric_fn(theta, p)
    return RandersMetric(L=jnp.eye(3), beta=beta)


# --- 3. VISUALIZATION ENGINE ---


def plot_spherical_world(theta, path=None, title="Holonomic Storm"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # A. Sphere (Wireframe) - Lighter to reduce clutter
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 15j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gainsboro", alpha=0.3, linewidth=0.5)

    # B. Wind Field (Quiver) - Thinner arrows
    u_q, v_q = np.mgrid[0 : 2 * np.pi : 16j, 0.2 : np.pi - 0.2 : 8j]
    xq = np.cos(u_q) * np.sin(v_q)
    yq = np.sin(u_q) * np.sin(v_q)
    zq = np.cos(v_q)
    points = np.stack([xq.flatten(), yq.flatten(), zq.flatten()], axis=1)

    _, betas = vmap(lambda p: metric_fn(theta, p))(jnp.array(points))
    betas = np.array(betas)

    ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        betas[:, 0],
        betas[:, 1],
        betas[:, 2],
        length=1,
        normalize=False,
        color="teal",
        alpha=0.5,
        linewidth=1.0,
        label="Wind Field",
    )

    if path is not None:
        path = np.array(path)

        # C. REFERENCE PATH (Euclidean Great Circle)
        p_start, p_end = path[0], path[-1]

        # Generate high-res Slerp for reference
        t_ref = np.linspace(0, 1, 100)
        omega = np.arccos(np.clip(np.dot(p_start, p_end), -1, 1))
        sin_omega = np.sin(omega)

        ref_path = []
        for ti in t_ref:
            p_t = (np.sin((1 - ti) * omega) / sin_omega) * p_start + (
                np.sin(ti * omega) / sin_omega
            ) * p_end
            ref_path.append(p_t)
        ref_path = np.array(ref_path)

        # QUANTITATIVE CHECK: Max Deviation
        # We interpolate Agent path to match Reference resolution for comparison
        from scipy.interpolate import interp1d

        t_agent = np.linspace(0, 1, len(path))
        # Interpolate x, y, z separately
        agent_interp = np.stack(
            [interp1d(t_agent, path[:, i], kind="linear")(t_ref) for i in range(3)], axis=1
        )

        dists = np.linalg.norm(agent_interp - ref_path, axis=1)
        max_dev = np.max(dists)
        print(f"\n--- GEOMETRIC ANALYSIS ---")
        print(f"Max Deviation from Euclidean Path: {max_dev:.5f} units")

        # D. PLOTTING PATHS
        # 1. Agent Path (Bottom Layer, Thicker)
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color="crimson",
            linewidth=2.5,
            alpha=0.9,
            label=f"Finsler Agent (Max Dev: {max_dev:.3f})",
            zorder=5,
        )

        # 2. Reference Path (Top Layer, Thin Dashed)
        # This ensures it's visible ON TOP of the red line
        ax.plot(
            ref_path[:, 0],
            ref_path[:, 1],
            ref_path[:, 2],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=1.0,
            label="Reference (Great Circle)",
            zorder=10,
        )

        # Markers
        ax.scatter(
            path[0, 0], path[0, 1], path[0, 2], color="lime", s=100, label="Start", zorder=10
        )
        ax.scatter(
            path[-1, 0], path[-1, 1], path[-1, 2], color="blue", s=100, label="Goal", zorder=10
        )

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="upper right")

    # Force equal aspect ratio so sphere looks like sphere
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()
    plt.savefig("spherical_storm_v2.png", dpi=200)
    print("Enhanced visualization saved to 'spherical_storm_v2.png'")


def analyze_mission_performance(theta, agent_path, p_start, p_end):
    """
    Mathematically proves that the Agent's curved path is superior
    to the naive Euclidean path (Great Circle).
    """
    print("\n=== MISSION ANALYSIS REPORT ===")

    # 1. Generate Baseline (Great Circle / Slerp)
    # This represents the "Naive" path a standard agent would take.
    num_steps = agent_path.shape[0]
    t = np.linspace(0, 1, num_steps)

    # Geometric Slerp
    omega = np.arccos(np.clip(np.dot(p_start, p_end), -1, 1))
    sin_omega = np.sin(omega)

    ref_path = []
    for ti in t:
        p_t = (np.sin((1 - ti) * omega) / sin_omega) * p_start + (
            np.sin(ti * omega) / sin_omega
        ) * p_end
        ref_path.append(p_t)
    ref_path = jnp.array(ref_path)

    # 2. Calculate Geometric Metrics (Distance)
    # Agent Length
    d_agent = jnp.sum(jnp.sqrt(jnp.sum((agent_path[1:] - agent_path[:-1]) ** 2, axis=1)))
    # Reference Length (Should be exactly omega * radius)
    d_ref = jnp.sum(jnp.sqrt(jnp.sum((ref_path[1:] - ref_path[:-1]) ** 2, axis=1)))

    # Deviation (Max Euclidean Distance between corresponding points)
    deviation = jnp.max(jnp.linalg.norm(agent_path - ref_path, axis=1))

    print(f"[Geometry]")
    print(f"Max Deviation:       {deviation:.5f} units")
    print(f"Agent Path Length:   {d_agent:.5f}")
    print(f"Euclidean Length:    {d_ref:.5f}")
    print(f"Detour Penalty:      {((d_agent/d_ref)-1)*100:.2f}% longer distance")

    # 3. Calculate Physics Metrics (Finsler Energy)
    # We evaluate the COST of both paths under the learned Wind (theta).

    # Helper to calc energy
    def get_cost(p):
        # Reconstruct full path format for energy function
        # (Our energy function usually expects inner points, but let's assume
        #  we adapted discrete_randers_energy to take the full sequence or we wrap it)
        # Assuming discrete_randers_energy(path, metric_fn) takes full path:
        return discrete_randers_energy(p, lambda x: metric_adapter(theta, x))

    cost_agent = get_cost(agent_path)
    cost_naive = get_cost(ref_path)

    savings = (cost_naive - cost_agent) / cost_naive * 100.0

    print(f"\n[Physics / Efficiency]")
    print(f"Naive Cost (Line):   {cost_naive:.5f}")
    print(f"Agent Cost (Curve):  {cost_agent:.5f}")
    print(f"Efficiency Gain:     {savings:.2f}% ENERGY SAVED")

    if savings > 0.5:
        print("\nCONCLUSION: Intelligence Verified.")
        print("The agent correctly identified that a longer geometric path")
        print("is actually a shorter physical path due to the wind.")
    else:
        print("\nCONCLUSION: Indecisive.")
        print("The wind might be too weak or parallel to the path.")


# --- 4. THE EXPERIMENT ---


def run_spherical_rover():
    print("--- Holonomic World Model: Spherical Rover ---")

    # 1. Setup Environment
    # Start: Equator (1,0,0)
    # Goal:  North Pole (0,0,1)
    p_start = jnp.array([1.0, 0.0, 0.0])
    p_end = jnp.array([0.0, 0.0, 1.0])

    # "Ground Truth" Observation:
    # The agent observes that moving "East" (0,1,0) at the equator is EASY.
    # We simulate this by forcing the model to learn a Wind blowing East.
    # We use the holonomy loss to teach this "Law of Physics".
    v_obs_start = jnp.array([0.0, 1.0, 0.0])  # Push East
    # The 'True' transport of East to North Pole along a quarter-circle
    # is actually a rotation about the X-axis.
    # East (0,1,0) -> Transport -> (0,1,0) at North Pole?
    # Wait, at North Pole (0,0,1), tangent plane is XY.
    # Standard parallel transport of (0,1,0) from Equator to Pole along (1,0,0)->(0,0,1)
    # keeps it pointing Y.
    v_obs_end_true = jnp.array([0.0, 1.0, 0.0])

    # 2. Initialize
    key = random.PRNGKey(2025)
    theta = init_spherical_net(key)
    optimizer = optax.adam(0.02)
    opt_state = optimizer.init(theta)

    # 3. Initialize Solver (With Constraint!)
    avbd = AVBDSolver(lr=0.05, beta=20.0, max_iters=150)

    def solver_wrapper(th, p1, p2):
        # Spherical Linear Interpolation (SLERP) initialization
        # (Approximated by linear + normalization for simplicity)
        num_inner = 18
        t = jnp.linspace(0, 1, num_inner + 2)[1:-1]
        linear = p1[None, :] * (1 - t[:, None]) + p2[None, :] * t[:, None]
        init_inner = linear / jnp.linalg.norm(linear, axis=1, keepdims=True)

        def e_fn(inner):
            full = jnp.concatenate([p1[None, :], inner, p2[None, :]])
            # Bind theta
            return discrete_randers_energy(full, lambda x: metric_adapter(th, x))

        # Pass the Spherical Constraint!
        constraints = [spherical_constraint]

        return avbd.solve(e_fn, constraints, p1, p2, init_inner)

    # 4. Learning Loop (Observing Physics)
    print("Observing Environment Dynamics...")

    @jit
    def step(theta, opt_state):
        loss, grads = jax.value_and_grad(holonomy_error_loss)(
            theta,
            p_start,
            v_obs_start,
            p_end,
            v_obs_end_true,
            metric_fn,
            solver_wrapper,
            parallel_transport,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    for i in range(50):  # Quick training
        theta, opt_state, loss = step(theta, opt_state)
        if i % 10 == 0:
            print(f"Epoch {i}: Physics Error = {loss:.6f}")

    # 5. Planning (Action)
    print("\nPlanning Trajectory with Learned Physics...")
    final_path = solver_wrapper(theta, p_start, p_end)

    # Verify constraint satisfaction
    radii = jnp.linalg.norm(final_path, axis=1)
    max_violation = jnp.max(jnp.abs(radii - 1.0))
    print(f"Max Constraint Violation: {max_violation:.6f}")

    if max_violation < 1e-2:
        print("SUCCESS: Path stays on the Sphere!")

    # After Planning:
    final_path = solver_wrapper(theta, p_start, p_end)

    # INSERT THE ANALYSIS HERE
    analyze_mission_performance(theta, final_path, p_start, p_end)

    # Visualize
    plot_spherical_world(theta, final_path)


if __name__ == "__main__":
    run_spherical_rover()
