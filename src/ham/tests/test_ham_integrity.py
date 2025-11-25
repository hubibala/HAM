import os
import sys

# Adjust path to find src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from ham.manifolds.sphere import Sphere
from ham.geometry.finsler import RandersFactory


def test_geometric_integrity():
    print("--- Testing HAM Geometric Integrity ---")

    # 1. Setup
    S2 = Sphere(dim=2)
    factory = RandersFactory(S2, epsilon=0.05)
    key = jax.random.PRNGKey(0)

    # 2. Generate Random Inputs (Simulating a Neural Net)
    batch_size = 10
    # Points on sphere
    x = S2.random_uniform(key, (batch_size,))

    # Raw outputs (Huge numbers to stress test bounds)
    raw_L = jax.random.normal(key, (batch_size, 3)) * 10.0
    raw_W = jax.random.normal(key, (batch_size, 3)) * 100.0  # Massive wind

    # 3. Construct Metric
    metric = factory.forward(x, raw_L, raw_W)

    print("Metric Constructed.")

    # 4. Verify Tangency
    # dot(x, beta) should be 0 for all items in batch
    radial_comp = jnp.sum(x * metric.beta, axis=-1)
    max_radial = jnp.max(jnp.abs(radial_comp))
    print(f"Max Radial Component (should be ~0): {max_radial:.2e}")
    assert max_radial < 1e-5, "Tangency constraint violated!"

    # 5. Verify Convexity
    # Check ||b||_{a^-1} < 1
    # norm^2 = b^T a^-1 b
    # Since a = L L^T, a^-1 = L^-T L^-1
    # norm = || L^-1 b ||

    # Invert L for check
    L_inv = jnp.linalg.inv(metric.L)

    # z = L^-1 b
    z = jnp.einsum("...ij,...j->...i", L_inv, metric.beta)
    z_norms = jnp.linalg.norm(z, axis=-1)

    max_norm = jnp.max(z_norms)
    print(f"Max Anisotropic Norm (should be < 1): {max_norm:.5f}")

    assert max_norm < 1.0, "Convexity constraint violated!"

    print("SUCCESS: HAM Framework is physically valid.")


if __name__ == "__main__":
    test_geometric_integrity()
