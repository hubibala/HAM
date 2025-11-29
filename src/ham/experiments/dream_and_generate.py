import os
import sys

# Force JAX to use CPU only (avoid Metal/StableHLO issues on Mac)
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad, vmap
import optax
import flax.linen as nn
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Any

# HAM Imports
from ham.models import RaceCarEncoder
from ham.geometry.finsler import RandersFactory, RandersMetric
from ham.solvers.avbd import AVBDSolver
from ham.geometry.transport import parallel_transport
from ham.manifolds import Sphere
from ham.sim.racecar import VanillaRaceCarWrapper


# Metric Network
class MetricNet(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, z) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(64)(z)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        raw_L = nn.Dense(self.latent_dim, kernel_init=nn.initializers.zeros)(x)
        raw_W = nn.Dense(self.latent_dim, kernel_init=nn.initializers.zeros)(x)
        return raw_L, raw_W


# Trainer
class GeometryTrainer:
    def __init__(self, latent_dim=3, lr=1e-3):
        self.model = MetricNet(latent_dim)
        self.manifold = Sphere(latent_dim)
        self.factory = RandersFactory(self.manifold, epsilon=0.05)
        self.tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

        model, factory, manifold, tx = self.model, self.factory, self.manifold, self.tx

        def train_step_logic(params, opt_state, safe_z, crash_z):
            def loss_fn(p):
                # 1. Barrier Loss
                out_crash = model.apply(p, crash_z)
                crash_metric = factory.forward(crash_z, *out_crash)
                crash_cost = jnp.trace(crash_metric.a, axis1=-2, axis2=-1)

                # Target Cost > 8.0 (Lowered from 20.0 to prevent exploding gradients)
                l_barrier = jnp.mean(jax.nn.relu(8.0 - crash_cost))

                # 2. Regularization (Identity Metric on Road)
                out_safe = model.apply(p, safe_z)
                safe_metric = factory.forward(safe_z, *out_safe)
                safe_cost = jnp.trace(safe_metric.a, axis1=-2, axis2=-1)
                l_reg = jnp.mean((safe_cost - 3.0) ** 2)

                return l_barrier + 0.1 * l_reg

            loss, grads = value_and_grad(loss_fn)(params)
            updates, new_opt = tx.update(grads, opt_state)
            return optax.apply_updates(params, updates), new_opt, loss

        self.train_step = jit(train_step_logic)

    def init_state(self, key, sample_input):
        params = self.model.init(key, sample_input)
        opt_state = self.tx.init(params)
        return params, opt_state


# Expert Generator
def generate_expert_data(key, metric_params, metric_net, factory, n_samples=1000):
    print("Initializing Solver...")
    # Low LR for stability on steep barriers
    solver = AVBDSolver(lr=0.01, max_iters=300)

    def energy_fn(params, path):
        L, W = metric_net.apply(params, path)
        metric = factory.forward(path, L, W)
        v = path[1:] - path[:-1]
        g = metric.a[:-1]
        beta = metric.beta[:-1]
        quad = jnp.einsum("ti,tij,tj->t", v, g, v)
        linear = jnp.einsum("ti,ti->t", beta, v)
        F = jnp.sqrt(jnp.maximum(quad, 1e-4)) + linear
        return jnp.sum(0.5 * F**2)

    k1, k2 = random.split(key)
    starts = random.normal(k1, (n_samples, 3))
    starts /= jnp.linalg.norm(starts, axis=1, keepdims=True)
    ends = random.normal(k2, (n_samples, 3))
    ends /= jnp.linalg.norm(ends, axis=1, keepdims=True)

    print(f"Solving {n_samples} Geodesics...")
    expert_states, expert_vectors = [], []
    failures = {"nan": 0, "err": 0}

    for i in tqdm(range(n_samples)):
        s, e = starts[i], ends[i]
        t = jnp.linspace(0, 1, 20)[1:-1, None]
        init_guess = s[None, :] * (1 - t) + e[None, :] * t
        init_guess /= jnp.linalg.norm(init_guess, axis=-1, keepdims=True)

        try:
            # This call matches the new AVBDSolver signature
            path = solver.solve(
                metric_params, energy_fn, [lambda x: jnp.sum(x**2) - 1.0], s, e, init_guess
            )

            if not jnp.all(jnp.isfinite(path)):
                failures["nan"] += 1
                continue

            vecs = path[1:] - path[:-1]
            vecs = vecs / (jnp.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6)

            expert_states.append(path[:-1])
            expert_vectors.append(vecs)
        except Exception as e:
            failures["err"] += 1
            if i == 0:
                print(f"Solver Error: {e}")
            pass

    print(f"Failures: {failures}")
    if not expert_states:
        return np.array([]), np.array([])
    return np.concatenate(expert_states), np.concatenate(expert_vectors)


# Main
if __name__ == "__main__":
    key = random.PRNGKey(42)

    print("Loading Encoder Params...")
    try:
        with open("encoder_params.pkl", "rb") as f:
            enc_params = pickle.load(f)
    except:
        print("ERROR: encoder_params.pkl not found. Run train_encoder.py first!")
        exit()

    encoder = RaceCarEncoder(latent_dim=3)

    print("Collecting Physics Data...")
    env = VanillaRaceCarWrapper(render_mode="rgb_array")
    obs, _ = env.reset()
    safe_frames, crash_frames = [], []

    for _ in tqdm(range(1000)):
        action = env.action_space.sample()
        action[1] = 0.5
        obs, reward, _, _, info = env.step(action)
        if info["crash"]:
            crash_frames.append(obs)
        else:
            safe_frames.append(obs)
    env.close()

    # Embed
    def embed(frames):
        if not frames:
            return jnp.zeros((0, 3))
        res = []
        for i in range(0, len(frames), 32):
            res.append(encoder.apply(enc_params, jnp.array(np.stack(frames[i : i + 32]))))
        return jnp.concatenate(res, axis=0)

    safe_z = embed(safe_frames[:200])
    crash_z = embed(crash_frames[:200])

    if len(crash_z) == 0:
        print("WARNING: No crashes found! Using synthetic barrier (South Pole).")
        crash_z = jnp.array([[0.0, 0.0, -1.0]])

    print("Training Geometry...")
    trainer = GeometryTrainer()
    params, opt = trainer.init_state(key, jnp.ones((1, 3)))

    losses = []
    for i in range(200):
        # Mini-batching
        idx_s = random.randint(random.fold_in(key, i), (32,), 0, len(safe_z))
        idx_c = random.randint(random.fold_in(key, i + 1), (32,), 0, len(crash_z))

        params, opt, loss = trainer.train_step(params, opt, safe_z[idx_s], crash_z[idx_c])
        losses.append(float(loss))
        if i % 50 == 0:
            print(f"Step {i}: Loss {loss:.4f}")

    print("Dreaming...")
    s, v = generate_expert_data(key, params, trainer.model, trainer.factory, n_samples=1000)

    print(f"Generated {len(s)} points.")
    with open("dream_data.pkl", "wb") as f:
        pickle.dump(
            {"states": s, "vectors": v, "enc_params": enc_params, "metric_params": params}, f
        )

    plt.plot(losses)
    plt.savefig("metric_loss.png")
    print("Done.")
