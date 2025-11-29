import os
import sys

# Force JAX to use CPU only (avoid Metal/StableHLO issues on Mac)
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# Imports
from ham.models import RaceCarEncoder
from ham.sim.racecar import IceRaceCarWrapper


def diagnose():
    print("--- HAM DIAGNOSTICS ---")

    # 1. Load Data
    try:
        with open("dream_data.pkl", "rb") as f:
            data = pickle.load(f)
        print("[OK] dream_data.pkl loaded.")
        enc_params = data.get("enc_params")
        if enc_params is None:
            print("[FAIL] 'enc_params' not found in dream_data.pkl")
            return
    except FileNotFoundError:
        print("[FAIL] dream_data.pkl not found.")
        return

    # 2. Analyze Expert Vectors (The "Dream")
    vectors = data["vectors"]
    mean_vec = jnp.mean(vectors, axis=0)
    std_vec = jnp.std(vectors, axis=0)

    print(f"\n1. Expert Vector Analysis (N={len(vectors)}):")
    print(f"   Mean Direction: {mean_vec}")
    print(f"   Std Deviation:  {std_vec}")

    # Check for Bias
    bias_mag = jnp.linalg.norm(mean_vec)
    if bias_mag > 0.8:
        print(
            f"   -> [FAIL] HIGH BIAS! The 'Expert' always points the same way (Mag={bias_mag:.2f})."
        )
        print("      Likely cause: Geometry has a single global gradient (Slope).")
    elif jnp.mean(std_vec) < 0.1:
        print("   -> [FAIL] LOW VARIANCE! The 'Expert' is ignoring context.")
    else:
        print("   -> [PASS] Vectors look diverse.")

    # 3. Analyze Encoder Separation (The "Eyes")
    print("\n2. Encoder Separation Analysis...")
    env = IceRaceCarWrapper(render_mode="rgb_array")
    encoder = RaceCarEncoder(latent_dim=3)

    obs, _ = env.reset()
    road_embeds = []
    grass_embeds = []

    print("   Collecting samples (Driving blindly)...")
    for _ in range(300):
        # Drive aggressively to hit grass
        action = env.action_space.sample()
        obs, _, _, _, info = env.step(action)

        # Embed
        z = encoder.apply(enc_params, jnp.array(obs)[None, ...])[0]

        # Basic heuristic: 'crash' usually means grass/off-road
        if info.get("crash", False):
            grass_embeds.append(z)
        else:
            road_embeds.append(z)

    env.close()

    if len(road_embeds) < 10 or len(grass_embeds) < 10:
        print(f"   [WARN] Insufficient data. Road: {len(road_embeds)}, Grass: {len(grass_embeds)}")
        return

    road_z = jnp.stack(road_embeds)
    grass_z = jnp.stack(grass_embeds)

    # Calculate Centroids
    road_center = jnp.mean(road_z, axis=0)
    grass_center = jnp.mean(grass_z, axis=0)

    # Normalize
    road_center /= jnp.linalg.norm(road_center)
    grass_center /= jnp.linalg.norm(grass_center)

    # Distance
    dist = jnp.linalg.norm(road_center - grass_center)

    print(f"\n   Stats:")
    print(f"   Road Center:  {road_center}")
    print(f"   Grass Center: {grass_center}")
    print(f"   Distance:     {dist:.4f} (Max possible on sphere is 2.0)")

    if dist < 0.2:
        print("   -> [CRITICAL FAIL] MODE COLLAPSE.")
        print("      The Encoder cannot distinguish Road from Grass.")
        print("      The Geometry Engine is trying to build a wall ON TOP of the road.")
    else:
        print("   -> [PASS] Encoder is separating concepts.")


if __name__ == "__main__":
    diagnose()
