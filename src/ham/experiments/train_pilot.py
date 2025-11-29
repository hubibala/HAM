import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import flax.linen as nn
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Any

# Imports
from ham.models import RaceCarEncoder
from ham.sim.racecar import VanillaRaceCarWrapper


# --- Models (Same as before) ---
class PilotNet(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(64)(z)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        v = nn.Dense(self.latent_dim)(x)
        v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
        return v


class MotorCortex(nn.Module):
    @nn.compact
    def __call__(self, z_current, z_target):
        x = jnp.concatenate([z_current, z_target], axis=-1)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        steer = nn.Dense(1)(x)
        steer = nn.tanh(steer)

        # Bias init for gas/brake
        gas_brake = nn.Dense(2, bias_init=lambda k, s, d: jnp.array([2.0, -5.0]))(x)
        gas_brake = nn.sigmoid(gas_brake)

        return jnp.concatenate([steer, gas_brake], axis=-1)


# --- Trainer ---
class AgentTrainer:
    def __init__(self, latent_dim=3, lr=1e-4):
        self.pilot = PilotNet(latent_dim)
        self.motor = MotorCortex()
        self.tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))

        pilot, motor = self.pilot, self.motor

        def total_loss_fn(params, z_t, z_next, a_true, z_dream, v_expert):
            a_pred = motor.apply(params["motor"], z_t, z_next)
            l_motor = jnp.mean((a_pred - a_true) ** 2)

            v_pred = pilot.apply(params["pilot"], z_dream)
            sim = jnp.sum(v_pred * v_expert, axis=-1)
            l_pilot = jnp.mean(1.0 - sim)
            return l_motor + l_pilot

        # JIT Update
        def train_step_logic(params, opt_state, motor_data, pilot_data):
            z_t, z_next, a_true = motor_data
            z_dream, v_expert = pilot_data
            val, grads = value_and_grad(total_loss_fn)(
                params, z_t, z_next, a_true, z_dream, v_expert
            )
            updates, new_opt = self.tx.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt, val

        self.train_step = jit(train_step_logic)

    def init_state(self, key, sample_z):
        k1, k2 = random.split(key)
        p_params = self.pilot.init(k1, sample_z)
        m_params = self.motor.init(k2, sample_z, sample_z)
        params = {"pilot": p_params, "motor": m_params}
        opt_state = self.tx.init(params)
        return params, opt_state


# --- Inference ---
def run_agent(env, encoder, enc_params, pilot, motor, agent_params):
    print("Running Inference...")
    obs, _ = env.reset()
    total_reward = 0
    encode = jit(lambda x: encoder.apply(enc_params, x))
    pilot_act = jit(lambda p, z: pilot.apply(p, z))
    motor_act = jit(lambda p, z, zt: motor.apply(p, z, zt))

    for t in range(500):
        obs_batch = jnp.array(obs[None, ...])
        z = encode(obs_batch)[0]
        v_desired = pilot_act(agent_params["pilot"], z)
        z_target = z + 0.5 * v_desired
        action = motor_act(agent_params["motor"], z, z_target)

        if np.isnan(action).any():
            action = np.array([0.0, 0.0, 0.0])
        else:
            action = np.array(action)

        if t % 20 == 0:
            s, g, b = action
            print(f"Step {t:03d} | S: {s:+.2f} G: {g:.2f} B: {b:.2f}")

        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        env.render()
        if term or trunc:
            break
    return total_reward


# --- Main ---
if __name__ == "__main__":
    key = random.PRNGKey(0)

    with open("dream_data.pkl", "rb") as f:
        dream_data = pickle.load(f)
    p_z = dream_data["states"]
    p_v = dream_data["vectors"]
    enc_params = dream_data["enc_params"]

    # --- FIX: Robust Data Collection ---
    print("Collecting Motor Calibration Data...")
    env = VanillaRaceCarWrapper(render_mode="rgb_array")
    encoder = RaceCarEncoder(latent_dim=3)

    motor_z_t, motor_z_next, motor_a = [], [], []
    obs, _ = env.reset()

    # Run longer loop
    pbar = tqdm(range(800))
    for i in pbar:
        action = env.action_space.sample()
        action[1] = np.random.uniform(0.6, 1.0)  # Gas
        action[2] = 0.0  # Brake
        action[0] = np.sin(i / 10.0)  # Steering

        next_obs, _, term, trunc, _ = env.step(action)

        # Skip Zoom-in phase (first 60 frames)
        if i > 60:
            diff = np.mean(np.abs(next_obs - obs))
            # Lower threshold + Skip check
            if diff > 0.005:
                z_t = encoder.apply(enc_params, jnp.array(obs)[None, ...])[0]
                z_next = encoder.apply(enc_params, jnp.array(next_obs)[None, ...])[0]
                motor_z_t.append(z_t)
                motor_z_next.append(z_next)
                motor_a.append(action)

            # Log progress
            if i % 50 == 0:
                pbar.set_description(f"Collected {len(motor_z_t)} samples")

        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    # Check count
    if len(motor_z_t) < 50:
        print(f"Only collected {len(motor_z_t)} samples.")
        raise ValueError("Car is STILL not moving! Check rendering setup.")

    m_z = jnp.stack(motor_z_t)
    m_zn = jnp.stack(motor_z_next)
    m_a = jnp.array(np.stack(motor_a))

    print(f"Success: Collected {len(m_z)} samples.")

    # Train
    print("Training Agent...")
    trainer = AgentTrainer(lr=2e-3)
    params, opt_state = trainer.init_state(key, m_z[0])

    for i in range(10000):
        params, opt_state, loss = trainer.train_step(
            params, opt_state, (m_z, m_zn, m_a), (p_z, p_v)
        )
        if i % 20 == 0:
            print(f"Epoch {i}: Loss {loss:.4f}")

    # Run
    print("Testing...")
    env = VanillaRaceCarWrapper(render_mode="human")
    reward = run_agent(env, encoder, enc_params, trainer.pilot, trainer.motor, params)
    print(f"Score: {reward}")
    env.close()
