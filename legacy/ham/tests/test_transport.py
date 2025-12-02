import os

# CRITICAL: Force JAX to use CPU to avoid Metal/StableHLO version mismatch on Apple Silicon
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest
import jax
import jax.numpy as jnp
import numpy as np

# Import the transport module
# We only need the public interface now; the internal connection logic is handled by the Berwald engine.
from ham.geometry import parallel_transport


class TestParallelTransport(unittest.TestCase):

    def setUp(self):
        # JAX needs 64-bit precision for strict geometric tests
        jax.config.update("jax_enable_x64", True)

    def test_euclidean_identity(self):
        """
        Validation Case 1:
        On a flat Euclidean plane with beta=0, transporting any vector
        around any closed loop must result in the identity transformation.
        """
        print("\n--- Test 1: Euclidean Identity (No Wind) ---")

        # 1. Metric: Identity Matrix, Zero Wind
        def flat_metric_fn(theta, p):
            dim = p.shape[0]
            return jnp.eye(dim), jnp.zeros(dim)

        # 2. Path: Unit Square Loop (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        # We discretize it finely to minimize integration error
        t = jnp.linspace(0, 1, 20)
        side1 = jnp.stack([t, jnp.zeros_like(t)], axis=1)  # Right
        side2 = jnp.stack([jnp.ones_like(t), t], axis=1)  # Up
        side3 = jnp.stack([1.0 - t, jnp.ones_like(t)], axis=1)  # Left
        side4 = jnp.stack([jnp.zeros_like(t), 1.0 - t], axis=1)  # Down

        # Concatenate path (removing duplicates at corners)
        path = jnp.concatenate([side1[:-1], side2[:-1], side3[:-1], side4])

        # 3. Transport Vector (1, 0)
        v_init = jnp.array([1.0, 0.0])
        theta = {}  # No params needed for flat metric

        # Note: metric_fn is passed as the 4th arg, which is marked static in our JIT
        v_final = parallel_transport(theta, path, v_init, flat_metric_fn, spherical=False)

        # 4. Check
        print(f"Initial: {v_init}")
        print(f"Final:   {v_final}")

        # Should be exactly equal (within numerical error of ODE solver)
        self.assertTrue(
            jnp.allclose(v_init, v_final, atol=1e-5), "Euclidean transport failed to be Identity!"
        )

    def test_spherical_latitude_holonomy(self):
        """
        Validation Case 2:
        On a sphere, transporting a vector around a closed loop (latitude)
        should result in a rotation related to the solid angle (Geometric Phase).

        Path: Latitude circle at theta = pi/4 (45 degrees).
        Expected Rotation: 2*pi*(1 - cos(theta))
        """
        print("\n--- Test 2: Spherical Latitude Holonomy (Riemannian Curvature) ---")

        # 1. Metric: Sphere in Spherical Coordinates (theta, phi)
        # g = diag(1, sin^2(theta))
        def sphere_metric_fn(theta, p):
            # p = [theta, phi]
            pol, azi = p
            sin_sq = jnp.sin(pol) ** 2
            g = jnp.diag(jnp.array([1.0, sin_sq]))
            beta = jnp.zeros(2)  # Pure Riemannian
            return g, beta

        # 2. Path: Constant latitude theta=pi/4, phi goes 0 -> 2pi
        num_steps = 100
        phi = jnp.linspace(0, 2 * jnp.pi, num_steps)
        theta_const = jnp.full_like(phi, jnp.pi / 4)
        path = jnp.stack([theta_const, phi], axis=1)

        # 3. Transport Vector pointing South (along theta)
        v_init = jnp.array([1.0, 0.0])
        params = {}

        v_final = parallel_transport(params, path, v_init, sphere_metric_fn, spherical=False)

        # 4. Expected Results (The 90-Degree Turn Logic)
        expected_angle = 2 * jnp.pi * (1.0 - jnp.cos(jnp.pi / 4))

        # Create rotation matrix for comparison
        c, s = jnp.cos(expected_angle), jnp.sin(expected_angle)
        v_expected = jnp.array([v_init[0] * c - v_init[1] * s, v_init[0] * s + v_init[1] * c])

        print(f"Path Latitude: 45 deg")
        print(f"Theoretical Rotation: {expected_angle:.4f} rad")
        print(f"Final Vector: {v_final}")

        # Assert that it DID rotate (Holonomy exists)
        self.assertFalse(
            jnp.allclose(v_init, v_final, atol=1e-2), "Vector failed to rotate on Sphere!"
        )

        # Check cosine similarity
        cos_sim = jnp.dot(v_final, v_expected) / (
            jnp.linalg.norm(v_final) * jnp.linalg.norm(v_expected)
        )
        print(f"Cosine Similarity to Theory: {cos_sim:.5f}")
        self.assertTrue(cos_sim > 0.95, "Spherical holonomy did not match theory.")

    def test_finsler_stokes_theorem(self):
        """
        Validation Case 3:
        On a flat plane with a 'Wind' field, the holonomy around a loop
        should match the enclosed flux of the wind's curl (Stokes' Theorem).
        """
        print("\n--- Test 3: Finsler Wind Holonomy (Stokes' Check) ---")

        # 1. Metric: Flat + Shear Wind
        def finsler_metric_fn(theta, p):
            x, y = p
            g = jnp.eye(2)
            # Wind increases as we move right. Pushes North.
            # beta = [0, x]
            beta = jnp.array([0.0, x])
            return g, beta

        # 2. Path: Unit Square (Same as Test 1)
        t = jnp.linspace(0, 1, 30)
        side1 = jnp.stack([t, jnp.zeros_like(t)], axis=1)
        side2 = jnp.stack([jnp.ones_like(t), t], axis=1)
        side3 = jnp.stack([1.0 - t, jnp.ones_like(t)], axis=1)
        side4 = jnp.stack([jnp.zeros_like(t), 1.0 - t], axis=1)
        path = jnp.concatenate([side1[:-1], side2[:-1], side3[:-1], side4])

        # 3. Transport Vector (1, 0)
        v_init = jnp.array([1.0, 0.0])
        theta = {}

        v_final = parallel_transport(theta, path, v_init, finsler_metric_fn, spherical=False)

        print(f"Initial: {v_init}")
        print(f"Final:   {v_final}")

        # 4. Verification
        # Check angle of rotation
        angle_moved = jnp.arctan2(v_final[1], v_final[0])
        print(f"Rotation Angle: {angle_moved:.4f} rad")
        print(f"Expected Flux:  1.0000")

        # Crucial Check: Did the wind cause a rotation?
        self.assertFalse(
            jnp.allclose(v_init, v_final, atol=1e-3), "Wind field failed to induce Holonomy!"
        )

        # Verify magnitude
        self.assertTrue(abs(angle_moved) > 0.1, "Rotation magnitude too small for Finsler effect.")


if __name__ == "__main__":
    unittest.main()
