import numpy as np
import gymnasium as gym
import cv2
import os

# Import wrapper for testing logic
from ham.sim.racecar import EgoRaceCarWrapper


def verify_cone_vision():
    print("--- VERIFYING EGO-VISION (AUTO-CENTERED) ---")

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    os.makedirs("vision_check", exist_ok=True)

    # We instantiate wrapper logic manually to debug
    wrapper = EgoRaceCarWrapper()
    wrapper.env = env

    obs, _ = env.reset()

    # --- AUTO-CALIBRATION ROUTINE ---
    print("Calibrating Rotation Center...")
    # Find Red Car Pixels (roughly R > 150, G < 100, B < 100)
    # Note: obs is 96x96x3
    mask_red = (obs[:, :, 0] > 180) & (obs[:, :, 1] < 100)
    y_idxs, x_idxs = np.where(mask_red)

    if len(y_idxs) > 0:
        center_y = int(np.mean(y_idxs))
        center_x = int(np.mean(x_idxs))
        # Manual tweak: The visual center of the red block might be slightly off the axle
        # Usually Box2D pivot is a bit further back than the visual center of the hood
        # Let's trust the visual center for now.
        rotation_center = (center_x, center_y)
        print(f"Detected Car Center at: {rotation_center}")
    else:
        print("WARNING: Car not found! Defaulting to (48, 70)")
        rotation_center = (48, 70)

    print("Driving... Check 'vision_check/' for debug images.")

    for i in range(100):
        # Drive in a circle to test rotation stability
        action = np.array([np.sin(i / 10.0), 0.3, 0.0])

        obs, reward, term, trunc, info = env.step(action)

        # 1. Physics Data
        car = env.unwrapped.car
        yaw = car.hull.angle

        # 2. Rotation Logic (using Calibrated Center)
        rows, cols = obs.shape[:2]

        # Angle: If car turns Left (Pos Yaw), Image tilts Left.
        # We must rotate Image Right (Neg Angle) to straighten it.
        angle_deg = -np.degrees(yaw)

        M = cv2.getRotationMatrix2D(rotation_center, angle_deg, 1.0)
        rotated_global = cv2.warpAffine(obs, M, (cols, rows), borderValue=(0, 0, 0))

        # 3. Crop Logic (Centered on the pivot)
        # We want 64x64. Pivot should be at the bottom-center of the crop.
        # Crop X: center_x - 32 to center_x + 32
        # Crop Y: center_y - 50 to center_y + 14 (Look ahead)

        cx, cy = rotation_center
        x1, x2 = cx - 32, cx + 32
        y1, y2 = cy - 54, cy + 10  # Look mostly ahead

        # Clamp bounds
        x1, x2 = max(0, x1), min(cols, x2)
        y1, y2 = max(0, y1), min(rows, y2)

        cone_view = rotated_global[y1:y2, x1:x2, :]

        # --- VISUALIZATION DASHBOARD ---
        if i % 5 == 0:
            # A. Global View (Show Pivot)
            global_vis = obs.copy()
            # Draw Pivot Crosshair
            cv2.line(global_vis, (cx - 5, cy), (cx + 5, cy), (0, 255, 0), 1)
            cv2.line(global_vis, (cx, cy - 5), (cx, cy + 5), (0, 255, 0), 1)

            # B. Rotated View (Show Crop)
            rotated_vis = rotated_global.copy()
            cv2.rectangle(rotated_vis, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue Box

            # C. Cone View (Resize for display)
            cone_vis = np.zeros((96, 96, 3), dtype=np.uint8)
            if cone_view.size > 0:
                # Center the crop in the frame
                h_c, w_c = cone_view.shape[:2]
                cone_vis[:h_c, :w_c] = cone_view
                # Draw "Up" vector
                cv2.arrowedLine(
                    cone_vis, (w_c // 2, h_c - 10), (w_c // 2, h_c - 40), (255, 255, 0), 2
                )

            # Debug Text
            cv2.putText(
                global_vis,
                f"Yaw: {np.degrees(yaw):.1f}",
                (5, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
            )
            cv2.putText(
                rotated_vis, "Rotated", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255)
            )
            cv2.putText(
                cone_vis, "Network View", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255)
            )

            combined = np.hstack([global_vis, rotated_vis, cone_vis])
            cv2.imwrite(
                f"vision_check/debug_{i:03d}.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            )

        if term or trunc:
            break

    env.close()
    print(f"Done. Detected Pivot: {rotation_center}")
    print(
        "Check 'vision_check/debug_050.png'. The car in the MIDDLE image should be perfectly vertical."
    )


if __name__ == "__main__":
    verify_cone_vision()
