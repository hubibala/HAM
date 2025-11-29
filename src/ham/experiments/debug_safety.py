import numpy as np
import gymnasium as gym
import cv2
import os
from ham.sim.racecar import EgoRaceCarWrapper


def debug_safety():
    print("--- DEBUGGING SAFETY SENSOR ---")
    os.makedirs("safety_debug", exist_ok=True)

    env = EgoRaceCarWrapper(render_mode="rgb_array")
    obs, _ = env.reset()

    print("Driving... Saving 'Danger' frames to 'safety_debug/'")

    count_safe = 0
    count_danger = 0

    for i in range(100):
        # Drive forward
        action = np.array([0.0, 0.5, 0.0])
        obs, reward, term, trunc, info = env.step(action)

        # --- FIX: Look at CURRENT FRAME only ---
        # Obs is (64, 64, 9). We want the last 3 channels.
        current_frame = obs[:, :, -3:]

        # Patch: Bottom Center of the Cone
        patch = current_frame[54:64, 27:37, :]

        # Green Logic: G > R+0.15 and G > B+0.15
        green_mask = (patch[:, :, 1] > patch[:, :, 0] + 0.15) & (
            patch[:, :, 1] > patch[:, :, 2] + 0.15
        )
        green_ratio = np.mean(green_mask)

        # Safety Score
        if green_ratio > 0.3:
            safety_score = 0.0  # Grass
        else:
            safety_score = 1.0  # Road

        # Check Info Flag
        crash_flag = info.get("crash", False)

        if crash_flag or safety_score < 0.5:
            label = "DANGER"
            count_danger += 1
            color = (0, 0, 255)  # Red text
        else:
            label = "SAFE"
            count_safe += 1
            color = (0, 255, 0)  # Green text

        # --- VISUALIZATION ---
        # Denormalize ONLY the current frame
        vis = (current_frame * 255).astype(np.uint8)

        # Now this works because shape is (64, 64, 3)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        # Draw stats
        text = f"{label} | Ratio: {green_ratio:.2f}"
        cv2.putText(vis, text, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Draw the patch box so you can see WHERE it is looking
        cv2.rectangle(vis, (27, 54), (37, 64), color, 1)

        cv2.imwrite(f"safety_debug/step_{i:03d}.png", vis)

        if term or trunc:
            break

    env.close()
    print(f"Stats: Safe={count_safe}, Danger={count_danger}")
    print("Check 'safety_debug/' images.")
    print("1. If the BOX is on gray road but says DANGER -> Increase threshold (+0.2).")
    print("2. If the BOX is on green grass but says SAFE -> Decrease threshold (+0.1).")


if __name__ == "__main__":
    debug_safety()
