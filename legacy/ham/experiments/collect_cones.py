import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pickle
import os
import cv2

# Use your Ego-Wrapper for the Cone logic
from ham.sim.racecar import EgoRaceCarWrapper


def collect_cone_data(num_steps=5000):
    print("Collecting Cone Vision Data...")
    env = EgoRaceCarWrapper(render_mode="rgb_array")
    obs, _ = env.reset()

    dataset = []

    for i in tqdm(range(num_steps)):
        # Behavior: Mostly drive straight, but occasionally swerve to find edges
        action = env.action_space.sample()
        action[1] = 0.4  # Gas
        action[0] = np.sin(i / 25.0)  # Smooth Wiggle to see all angles of road

        # Step
        # info['crash'] comes from the wrapper's green detection
        next_obs, _, term, trunc, info = env.step(action)

        # --- REWARD LOGIC (The "Value" of the state) ---
        # We want to assign a scalar "Safety" score to this image.
        # 1.0 = Perfect Road
        # 0.0 = Grass / Crash

        # Heuristic: Check Green Channel vs Red/Blue in the Cone
        # Cone is 64x64. Let's look at the bottom half (closest to car)
        patch = obs[32:, :, :]

        # Count green pixels
        # Green if G > R+10 and G > B+10
        green_mask = (patch[:, :, 1] > patch[:, :, 0] + 0.1) & (
            patch[:, :, 1] > patch[:, :, 2] + 0.1
        )
        green_ratio = np.mean(green_mask)

        # Safety Score: 1.0 if no green, approaches 0.0 as green fills view
        safety_score = 1.0 - np.clip(green_ratio * 2.0, 0.0, 1.0)

        # Also punish explicitly flagged crashes
        if info.get("crash", False):
            safety_score = 0.0

        dataset.append(
            {"image": obs, "safety": safety_score}  # 64x64x9 Cone Stack  # Scalar [0, 1]
        )

        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    env.close()

    # Analyze Distribution
    safeties = [d["safety"] for d in dataset]
    print(f"Collected {len(dataset)} samples.")
    print(f"Safety Stats: Mean={np.mean(safeties):.2f}, Min={np.min(safeties):.2f}")
    print(f"Safe (>0.9): {np.sum(np.array(safeties)>0.9)}")
    print(f"Danger (<0.1): {np.sum(np.array(safeties)<0.1)}")

    with open("cone_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print("Saved cone_dataset.pkl")


if __name__ == "__main__":
    collect_cone_data()
