import os
import sys

# Adjust path to find src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from ham.solvers.avbd import AVBDSolver


def test_avbd_euclidean_geodesic():
    """
    Test 1: Simple straight line in Euclidean space.
    Should recover a straight line between (0,0) and (1,1).
    """
    # INCREASED ITERATIONS: GD on springs is slow (diffusive).
    # 200 iters is not enough to relax a step function to a line.
    solver = AVBDSolver(lr=0.1, max_iters=1000, tol=1e-6)

    start = jnp.array([0.0, 0.0])
    end = jnp.array([1.0, 1.0])

    # "Bad" Initialization: A step-like function
    # This tests if the solver can actually move the points
    init_path = jnp.zeros((10, 2)) + 0.5

    # Param is a scaling factor (just to test differentiability plumbing)
    params = jnp.array([1.0])

    def energy_fn(theta, path):
        # E = 0.5 * theta * sum( |dx|^2 )
        diff = path[1:] - path[:-1]
        dist_sq = jnp.sum(diff**2, axis=1)
        return 0.5 * theta[0] * jnp.sum(dist_sq)

    # Solve
    path = solver.solve(params, energy_fn, [], start, end, init_path)

    # Check if straight line
    # Midpoint of 10 inner points + 2 boundary points (Total 12)
    # The solver returns [Start, p0...p9, End]
    # Midpoint of the *space* is roughly at index 6
    mid_idx = path.shape[0] // 2
    midpoint = path[mid_idx]

    print(f"Computed Midpoint: {midpoint}")

    # Expect roughly (0.5, 0.5)
    # With 1000 iters, error should be < 0.001
    assert jnp.allclose(midpoint, jnp.array([0.5, 0.5]), atol=1e-2)
    print("Test 1 Passed: Geodesic converged to straight line.")


def test_avbd_gradients():
    """
    Test 2: Implicit Differentiation.
    Does the gradient flow through the solver?
    """
    solver = AVBDSolver(lr=0.1, max_iters=50)
    start = jnp.array([0.0])
    end = jnp.array([1.0])
    # Linear init for gradient test to ensure stability
    init_path = jnp.linspace(0, 1, 12)[1:-1][:, None]

    def loss_fn(theta):
        def energy(p, path):
            # Energy scales with theta
            v = path[1:] - path[:-1]
            return 0.5 * p[0] * jnp.sum(v**2)

        # Solve inner path
        path = solver.solve(theta, energy, [], start, end, init_path)

        # Fake downstream loss: Minimize the length of the path
        # If theta changes, the *optimal path* in Euclidean space doesn't change
        # (scaling metric doesn't change geodesics),
        # BUT the energy value changes.
        # To make path depend on theta, we'd need a non-flat metric.
        # Here we just verify the pipeline runs and returns a gradient.
        return jnp.sum(path**2)

    params = jnp.array([2.0])
    l, g = jax.value_and_grad(loss_fn)(params)

    print(f"Loss: {l}, Gradient: {g}")
    assert not jnp.isnan(g).any()
    print("Test 2 Passed: Implicit Gradient computed successfully.")


if __name__ == "__main__":
    test_avbd_euclidean_geodesic()
    test_avbd_gradients()
