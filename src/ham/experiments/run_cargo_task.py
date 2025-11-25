import os
import sys

# Force JAX to use CPU only (avoid Metal/StableHLO issues on Mac)
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, grad
import optax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- THE FULL STACK ---
from ham.sim import coriolis_env
from ham.sim.coriolis_env import EnvParams, EnvState
from ham.models import encoder
from ham.solvers import AVBDSolver
from ham.geometry import parallel_transport
from ham.losses import holonomy_error_loss
from ham.experiments.teleportation_avbd import (
    discrete_randers_energy,
    metric_adapter,
    RandersMetric,
)

# === CONFIGURATION ===
LATENT_DIM = 3  # <--- Change this to 4, 16, or 128 to test scalability
# =====================


# --- 1. DATA COLLECTION (Unchanged) ---
def collect_experience(key, room_id, num_episodes=50, steps=20):
    # ... (Same as previous) ...
    print(f"Collecting observation data from Room {room_id}...")
    params = EnvParams()

    def run_episode(k):
        state = coriolis_env.reset(k, room_id=room_id, params=params)

        def step_fn(s, _):
            next_s = coriolis_env.step(s, jnp.zeros(2), params)
            img = coriolis_env.render(next_s, params)
            return next_s, img

        _, imgs = jax.lax.scan(step_fn, state, None, length=steps)
        return imgs

    keys = random.split(key, num_episodes)
    batch_imgs = vmap(run_episode)(keys)
    return batch_imgs


# --- 2. METRIC NETWORK (Parametrized) ---


def init_metric_net(key, dim=LATENT_DIM):
    # Input: dim -> Output: dim (Wind Vector)
    return {
        "w1": random.normal(key, (dim, 64)) * 0.1,
        "b1": jnp.zeros(64),
        "w2": random.normal(key, (64, dim)) * 0.1,
        "b2": jnp.zeros(dim),
    }


def metric_fn(theta, p):
    # 1. Dynamic Dimension
    dim = p.shape[-1]
    g = jnp.eye(dim)

    # 2. Wind Network
    h = jnp.tanh(jnp.dot(p, theta["w1"]) + theta["b1"])
    raw_wind = jnp.dot(h, theta["w2"]) + theta["b2"]

    # 3. Project to Tangent Space of S^n
    normal = p / (jnp.linalg.norm(p) + 1e-6)
    wind_tangent = raw_wind - jnp.dot(raw_wind, normal) * normal

    # 4. Soft saturation
    beta = 0.9 * jnp.tanh(wind_tangent)
    return g, beta


# --- 3. TRAINING LOOPS ---


def train_perception(key, data_A, data_B):
    print(f"\n--- Phase 1: Training Perception (Dim={LATENT_DIM}) ---")
    all_data = jnp.concatenate([data_A, data_B], axis=0)

    # Init Encoder with correct dim
    enc_params = encoder.init_encoder_params(key, output_dim=LATENT_DIM)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(enc_params)

    @jit
    def step(params, opt, batch):
        loss, grads = jax.value_and_grad(encoder.contrastive_loss)(params, batch)
        updates, opt = optimizer.update(grads, opt)
        return optax.apply_updates(params, updates), opt, loss

    for i in range(100):
        idx = random.randint(random.fold_in(key, i), (32,), 0, all_data.shape[0])
        batch = all_data[idx]
        enc_params, opt_state, loss = step(enc_params, opt_state, batch)
        if i % 20 == 0:
            print(f"Encoder Epoch {i}: Loss = {loss:.4f}")

    return enc_params


def train_physics(key, enc_params, data_A, data_B):
    print("\n--- Phase 2: Learning Latent Physics ---")

    def encode_traj(traj):
        T = traj.shape[0]
        flat = traj.reshape(T, 64, 64, 1)
        zs = encoder.apply_encoder(enc_params, flat)
        return zs

    z_A = vmap(encode_traj)(data_A)
    z_B = vmap(encode_traj)(data_B)

    # Init Metric with correct dim
    theta = init_metric_net(key, dim=LATENT_DIM)
    optimizer = optax.adam(5e-3)
    opt_state = optimizer.init(theta)

    @jit
    def physics_loss(theta, batch_z):
        def path_energy(z_path):
            return discrete_randers_energy(z_path, lambda p: metric_adapter(theta, p))

        return jnp.mean(vmap(path_energy)(batch_z))

    @jit
    def step(theta, opt, batch):
        loss, grads = jax.value_and_grad(physics_loss)(theta, batch)
        updates, opt = optimizer.update(grads, opt)
        return optax.apply_updates(theta, updates), opt, loss

    all_z = jnp.concatenate([z_A, z_B], axis=0)

    for i in range(100):
        idx = random.randint(random.fold_in(key, i), (64,), 0, all_z.shape[0])
        batch = all_z[idx]
        theta, opt_state, loss = step(theta, opt_state, batch)
        if i % 20 == 0:
            print(f"Physics Epoch {i}: Loss = {loss:.4f}")

    return theta, z_A, z_B


# --- 4. CALIBRATION & TRANSFER ---


def calibrate_and_transfer(enc_params, theta, z_A, z_B):
    print("\n--- Phase 3: Holonomic Calibration & Transfer ---")

    center_A = jnp.mean(z_A[:, 0, :], axis=0)
    center_A /= jnp.linalg.norm(center_A)

    center_B = jnp.mean(z_B[:, 0, :], axis=0)
    center_B /= jnp.linalg.norm(center_B)

    _, beta_A = metric_fn(theta, center_A)
    _, beta_B = metric_fn(theta, center_B)

    # Print magnitude instead of full vector if dim is huge
    print(f"Gravity A Magnitude: {jnp.linalg.norm(beta_A):.4f}")

    print("Calibrating geometry...")
    calib_optimizer = optax.adam(0.01)
    calib_state = calib_optimizer.init(theta)
    avbd = AVBDSolver(lr=0.05, beta=10.0, max_iters=50)

    def solver_wrapper(th, p1, p2):
        t = jnp.linspace(0, 1, 10)
        init = p1[None, :] * (1 - t[:, None]) + p2[None, :] * t[:, None]
        init = init / jnp.linalg.norm(init, axis=1, keepdims=True)

        def e_fn(inner):
            full = jnp.concatenate([p1[None, :], inner, p2[None, :]])
            return discrete_randers_energy(full, lambda x: metric_adapter(th, x))

        return avbd.solve(e_fn, [lambda x: jnp.sum(x**2) - 1.0], p1, p2, init)

    @jit
    def calibration_step(th, opt):
        loss, grads = jax.value_and_grad(holonomy_error_loss)(
            th, center_A, beta_A, center_B, beta_B, metric_fn, solver_wrapper, parallel_transport
        )
        updates, opt = calib_optimizer.update(grads, opt)
        return optax.apply_updates(th, updates), opt, loss

    for i in range(50):
        theta, calib_state, c_loss = calibration_step(theta, calib_state)
        if i % 10 == 0:
            print(f"Calibration Error: {c_loss:.4f}")

    print("\n--- Phase 4: Zero-Shot Transfer ---")
    v_skill_A = -beta_A
    target_dir = -beta_B

    transfer_path = solver_wrapper(theta, center_A, center_B)
    v_transported = parallel_transport(theta, transfer_path, v_skill_A, metric_fn)

    cos_sim = jnp.dot(v_transported, target_dir) / (
        jnp.linalg.norm(v_transported) * jnp.linalg.norm(target_dir)
    )
    print(f"Skill Transfer Cosine Similarity: {cos_sim:.4f}")

    return {
        "z_A": z_A,
        "z_B": z_B,
        "center_A": center_A,
        "center_B": center_B,
        "beta_A": beta_A,
        "beta_B": beta_B,
        "v_skill_A": v_skill_A,
        "v_transported": v_transported,
        "v_target": target_dir,
    }


# --- 5. VISUALIZATION (High-Dim Compatible) ---


def visualize_final_result(res):
    z_A, z_B = res["z_A"], res["z_B"]
    dim = z_A.shape[-1]

    # If High-Dim, project to 3D using PCA logic (SVD)
    if dim > 3:
        print(f"\n[Visualization] Projecting {dim}D -> 3D for plotting...")
        # Collect all points to find principal components
        all_points = np.concatenate([z_A.reshape(-1, dim), z_B.reshape(-1, dim)], axis=0)
        # Simple SVD
        u, s, vh = np.linalg.svd(all_points, full_matrices=False)
        projection_matrix = vh[:3, :].T  # (dim, 3)

        # Project data
        def proj(x):
            return np.dot(x, projection_matrix)

        p_A = proj(res["center_A"])
        p_B = proj(res["center_B"])
        vec_A = proj(res["v_skill_A"])
        vec_trans = proj(res["v_transported"])
        vec_targ = proj(res["v_target"])

        # Normalize projected vectors for visualization
        vec_A /= np.linalg.norm(vec_A)
        vec_trans /= np.linalg.norm(vec_trans)
        vec_targ /= np.linalg.norm(vec_targ)

        # Project clusters (subsample)
        cluster_A = proj(z_A[:100, 0, :])
        cluster_B = proj(z_B[:100, 0, :])

    else:
        p_A, p_B = res["center_A"], res["center_B"]
        vec_A, vec_trans, vec_targ = res["v_skill_A"], res["v_transported"], res["v_target"]
        cluster_A, cluster_B = z_A[:100, 0, :], z_B[:100, 0, :]

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw Sphere (Wireframe)
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    ax.scatter(cluster_A[:, 0], cluster_A[:, 1], cluster_A[:, 2], color="blue", alpha=0.3)
    ax.scatter(cluster_B[:, 0], cluster_B[:, 1], cluster_B[:, 2], color="orange", alpha=0.3)

    # Vectors
    ax.quiver(
        p_A[0],
        p_A[1],
        p_A[2],
        vec_A[0],
        vec_A[1],
        vec_A[2],
        color="cyan",
        length=1,
        linewidth=3,
        label="Skill A",
    )
    ax.quiver(
        p_B[0],
        p_B[1],
        p_B[2],
        vec_trans[0],
        vec_trans[1],
        vec_trans[2],
        color="magenta",
        length=1,
        linewidth=3,
        label="Transported",
    )
    ax.quiver(
        p_B[0],
        p_B[1],
        p_B[2],
        vec_targ[0],
        vec_targ[1],
        vec_targ[2],
        color="green",
        length=1,
        linewidth=2,
        linestyle="--",
        label="Target",
    )

    ax.set_title(f"Transfer (Dim: {dim})")
    ax.legend()
    plt.savefig("coriolis_calibrated.png", dpi=150)
    print("Saved plot.")


if __name__ == "__main__":
    key = random.PRNGKey(2025)
    data_A = collect_experience(key, room_id=0)
    data_B = collect_experience(key, room_id=1)
    enc_params = train_perception(key, data_A, data_B)
    theta, z_A, z_B = train_physics(key, enc_params, data_A, data_B)
    res = calibrate_and_transfer(enc_params, theta, z_A, z_B)
    visualize_final_result(res)
