import gymnasium as gym
import numpy as np
import cv2
from collections import deque
from typing import Any


class EgoRaceCarWrapper(gym.Wrapper):
    """
    Ego-Centric Wrapper.
    Rotates the world so the car always faces UP (North).
    Crops a 64x64 'Cone of Vision' ahead of the car.
    """

    def __init__(self, render_mode="rgb_array"):
        env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=True)
        super().__init__(env)
        self.frame_stack = deque(maxlen=3)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # On reset, car is at 0 yaw
        proc_obs = self._process_obs(obs, 0.0)
        for _ in range(3):
            self.frame_stack.append(proc_obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Get Car Yaw
        car = self.env.unwrapped.car
        yaw = car.hull.angle if car else 0.0

        # 2. Process: Rotate & Crop
        proc_obs = self._process_obs(obs, yaw)

        # 3. Detect Surface (Center of the cropped cone)
        # We look at the bottom center of the crop (immediate front)
        # Crop is 64x64. Bottom center is roughly y=54-64, x=27-37
        patch = proc_obs[54:64, 27:37, :]

        # Grass check (Green dominant)
        # Normalized pixel values 0-1
        is_grass = np.mean(patch[:, :, 1]) > np.mean(patch[:, :, 0]) + 0.5
        info["crash"] = is_grass or (reward < -0.1)

        self.frame_stack.append(proc_obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _process_obs(self, obs, yaw):
        # Input: 96x96x3 (Global)
        rows, cols = obs.shape[:2]

        # CALIBRATED CENTER (From verify_vision.py)
        # Adjust this if your verification showed something different!
        center = (48, 66)

        # Inverse Rotation to stabilize world relative to car
        angle_deg = -np.degrees(yaw)

        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(obs, M, (cols, rows), borderValue=(0, 0, 0))

        # Crop 64x64 centered on the pivot
        cx, cy = center
        x1, x2 = cx - 32, cx + 32
        y1, y2 = cy - 54, cy + 10  # Look ahead

        # Clamp to image bounds
        x1, x2 = max(0, x1), min(cols, x2)
        y1, y2 = max(0, y1), min(rows, y2)

        crop = rotated[y1:y2, x1:x2, :]

        # Handle edge case where rotation/crop goes out of bounds
        if crop.shape[0] != 64 or crop.shape[1] != 64:
            crop = cv2.resize(crop, (64, 64))

        return crop.astype(np.float32) / 255.0

    def _get_stacked_obs(self):
        return np.concatenate(self.frame_stack, axis=-1)
