import os
import sys

# 1. Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, "../../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from jax import value_and_grad, vmap

# --- IMPORTS ---
from ham.manifolds import Sphere
from ham.geometry import RandersFactory, discrete_randers_energy
from ham.nn import MetricNet
from ham.solvers import ProjectedGradientSolver, AVBDSolver


def main():
    print("--- Reproducing: Holonomic Control (Better View + Stronger Wind) ---")

    manifold = Sphere(dim=2)

    # TUNING 1: Allow stronger winds (99% speed of light)
    # This gives the solver more "control authority" to bend the path.
    factory = RandersFactory(manifold, epsilon=0.01)

    key = jax.random.PRNGKey(42)
    net = MetricNet(key, input_dim=3, output_dim=3, hidden_dim=64)

    # Solver
    inner_solver = ProjectedGradientSolver(manifold, lr=0.005, max_iters=600)

    # Task
    start_p = jnp.array([1.0, 0.0, 0.0])
    end_p = jnp.array([0.0, 1.0, 0.0])
    waypoint_p = jnp.array([0.0, 0.0, 1.0])

    # Init Path
    steps = 20
    t = jnp.linspace(0, 1, steps)[1:-1]
    init_inner = start_p[None, :] * (1 - t[:, None]) + end_p[None, :] * t[:, None]
    init_inner = vmap(manifold.projection)(init_inner)

    # Partition
    params, static = eqx.partition(net, eqx.is_array)

    # TUNING 2: Slower, Safer Learning Rate
    # We are playing with fire (epsilon=0.01), so we move slowly.
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.005))  # Reduced from 0.01
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def train_step(params, static, opt_state):

        def loss_fn(p):
            model = eqx.combine(p, static)

            def metric_fn(x):
                raw_L, raw_W = model(x)
                return factory.forward(x, raw_L, raw_W)

            final_path = inner_solver.solve(
                lambda p: discrete_randers_energy(p, metric_fn), start_p, end_p, init_inner
            )

            dists = jnp.linalg.norm(final_path - waypoint_p[None, :], axis=1)
            min_dist = jnp.min(dists)

            # Minimal Reg
            reg = 1e-6 * sum(jnp.sum(w**2) for w in jax.tree_util.tree_leaves(p))

            return min_dist + reg, final_path

        (loss, path), grads = value_and_grad(loss_fn, has_aux=True)(params)

        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss, path

    print("Training started...")

    best_params = params
    best_loss = 100.0

    for epoch in range(401):  # More epochs for slow LR
        new_params, opt_state, loss, final_path = train_step(params, static, opt_state)

        loss_val = float(loss)
        if np.isnan(loss_val):
            print(f"WARNING: NaN detected at epoch {epoch}. Reverting to best.")
            params = best_params
            break

        params = new_params
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Waypoint Miss: {loss:.4f}")

    # Visualization
    final_net = eqx.combine(best_params, static)

    print("Generating Visualization (Rotated View)...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Better Sphere Grid
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    # Re-solve
    def metric_fn(x):
        raw_L, raw_W = final_net(x)
        return factory.forward(x, raw_L, raw_W)

    viz_path = inner_solver.solve(
        lambda p: discrete_randers_energy(p, metric_fn), start_p, end_p, init_inner
    )

    path_np = np.array(viz_path)
    ax.plot(
        path_np[:, 0], path_np[:, 1], path_np[:, 2], "r-", linewidth=4, label="Learned Geodesic"
    )

    ax.scatter(1, 0, 0, c="g", s=150, label="Start")
    ax.scatter(0, 1, 0, c="b", s=150, label="End")
    ax.scatter(0, 0, 1, c="gold", s=250, marker="*", label="Target")

    # Wind Field
    def get_wind(p):
        raw_L, raw_W = final_net(p)
        m = factory.forward(p, raw_L, raw_W)
        return -m.beta

    grid_pts = []
    # Show wind in the relevant octant
    for t in np.linspace(0, np.pi / 2, 8):
        for p in np.linspace(0, np.pi / 2, 8):
            grid_pts.append([np.cos(t) * np.sin(p), np.sin(t) * np.sin(p), np.cos(p)])
    grid_pts = jnp.array(grid_pts)

    winds = vmap(get_wind)(grid_pts)

    ax.quiver(
        grid_pts[:, 0],
        grid_pts[:, 1],
        grid_pts[:, 2],
        winds[:, 0],
        winds[:, 1],
        winds[:, 2],
        length=1,
        color="cyan",
        label="Learned Drift",
    )

    # TUNING 3: Optimal Camera Angle
    # Azimuth 45 (between X and Y axes) looks at the "front" of the bend
    # Elevation 30 looks down slightly
    ax.view_init(elev=30, azim=45)

    ax.set_title("Figure 1: Holonomic Control (Side View)")
    ax.legend()
    plt.savefig("reproduction_result.png", dpi=150)
    print("Success! Saved reproduction_result.png")


if __name__ == "__main__":
    main()
