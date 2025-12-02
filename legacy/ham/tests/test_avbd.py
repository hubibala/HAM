import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from ham.manifolds import Sphere
from ham.solvers import AVBDSolver


def test_avbd_constraint_hardening():
    print("\n--- Testing AVBD Solver (Constraint Hardening) ---")

    # 1. Setup Problem
    start = jnp.array([1.0, 0.0, 0.0])
    end = jnp.array([0.0, 1.0, 0.0])

    # Initial Path (Straight line through the sphere)
    # This VIOLATES the manifold constraint ||x||=1 significantly (midpoint norm ~0.7)
    t = jnp.linspace(0, 1, 10)[1:-1]
    init_guess = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]

    # 2. Define Energies
    # Physics: Just minimize path length (Euclidean for simplicity)
    def energy_fn(full_path):
        diffs = full_path[1:] - full_path[:-1]
        return 0.5 * jnp.sum(diffs**2)

    # Constraints: Manifold Constraint C(x) = ||x|| - 1 = 0
    # AVBD must drive this to 0.
    def manifold_constraint(x):
        return jnp.linalg.norm(x) - 1.0

    # 3. Run Solver
    # Low LR because stiffness gets huge
    solver = AVBDSolver(lr=0.005, beta=5.0, max_iters=500)

    final_path = solver.solve(energy_fn, [manifold_constraint], start, end, init_guess)

    # 4. Validation
    norms = jnp.linalg.norm(final_path, axis=1)
    max_violation = jnp.max(jnp.abs(norms - 1.0))

    print(f"Initial Midpoint Norm: {jnp.linalg.norm(init_guess[4]):.4f}")
    print(f"Final Midpoint Norm:   {norms[5]:.4f}")
    print(f"Max Constraint Violation: {max_violation:.6f}")

    # It won't be 0.000000 like projection, but should be small (< 1e-2)
    assert max_violation < 0.02, "AVBD failed to enforce hard constraint!"
    print("SUCCESS: AVBD learned to stay on the manifold via constraints.")


if __name__ == "__main__":
    test_avbd_constraint_hardening()
