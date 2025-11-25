import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from ham.manifolds import Sphere
from ham.geometry import RandersFactory, discrete_randers_energy
from ham.solvers import ProjectedGradientSolver


def test_solver_physics():
    print("\n--- Testing Solver & Physics Engine ---")

    manifold = Sphere(dim=2)
    factory = RandersFactory(manifold, epsilon=0.05)
    solver = ProjectedGradientSolver(manifold, lr=0.05, max_iters=200)

    start = jnp.array([1.0, 0.0, 0.0])
    end = jnp.array([0.0, 1.0, 0.0])

    # Initial Guess
    t = jnp.linspace(0, 1, 10)[1:-1]
    init_guess = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]

    # --- TEST CASE A: Riemannian ---
    print("\n[Case A] Riemannian Sphere (Zero Wind)")

    def riemannian_metric_fn(x):
        raw_L = jnp.zeros(3) + 0.54
        raw_W = jnp.zeros(3)
        return factory.forward(x, raw_L, raw_W)

    path_r = solver.solve(
        lambda p: discrete_randers_energy(p, riemannian_metric_fn), start, end, init_guess
    )

    E_r = discrete_randers_energy(path_r, riemannian_metric_fn)
    print(f"Riemannian Energy: {E_r:.4f}")

    # --- TEST CASE B: Finsler (Headwind) ---
    print("\n[Case B] Finsler Sphere (Headwind)")

    def finsler_metric_fn(x):
        # FIX: Align raw_W WITH the path direction to create Resistance (Beta > 0)
        # Path goes roughly [-1, 1, 0]. We set W to match.
        raw_L = jnp.zeros(3) + 0.54
        raw_W = jnp.array([-1.0, 1.0, 0.0]) * 5.0  # Aligned with motion
        return factory.forward(x, raw_L, raw_W)

    path_f = solver.solve(
        lambda p: discrete_randers_energy(p, finsler_metric_fn), start, end, init_guess
    )

    E_f = discrete_randers_energy(path_f, finsler_metric_fn)
    print(f"Finsler Energy (Headwind): {E_f:.4f}")

    # Compare Energy of the SAME STRAIGHT PATH under both metrics
    # This isolates the metric effect from the solver's path change
    E_r_on_r = discrete_randers_energy(path_r, riemannian_metric_fn)
    E_f_on_r = discrete_randers_energy(path_r, finsler_metric_fn)

    print(f"Straight Path Cost (Riemannian): {E_r_on_r:.4f}")
    print(f"Straight Path Cost (Finsler):    {E_f_on_r:.4f}")

    # Now E_f should be huge because we are fighting the beta term
    assert E_f_on_r > E_r_on_r, f"Headwind failed! {E_f_on_r} should be > {E_r_on_r}"
    print("SUCCESS: Physics Engine respects wind resistance.")


if __name__ == "__main__":
    test_solver_physics()
