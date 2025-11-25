import jax
import jax.numpy as jnp
from jax import vmap
from typing import Callable
from .finsler import RandersMetric


def discrete_randers_energy(
    path: jnp.ndarray, metric_fn: Callable[[jnp.ndarray], RandersMetric]
) -> jnp.float32:
    """
    Calculates the discrete Randers energy of a path.
    Action = Sum 0.5 * F(x, v)^2

    Args:
        path: Shape (Steps, Dim) - The sequence of points.
        metric_fn: A function f(x) -> RandersMetric.
                   This is usually the Neural Network + Factory.
    """
    # 1. Compute Velocities and Midpoints
    # v_t = x_{t+1} - x_t
    velocities = path[1:] - path[:-1]

    # We evaluate the metric at the midpoint of the segment
    midpoints = (path[1:] + path[:-1]) / 2.0

    # 2. Define Step Energy
    def step_energy(v, x):
        m = metric_fn(x)

        # F(v) = sqrt(v.T a v) + b.T v
        # Note: m.L is Cholesky of 'a'
        # v.T a v = || L.T v ||^2
        Lv = jnp.dot(m.L.T, v)

        # Riemannian part (alpha)
        alpha = jnp.sqrt(jnp.dot(Lv, Lv) + 1e-9)

        # Drift part (beta)
        drift = jnp.dot(m.beta, v)

        # Randers Squared Norm (Lagrangian)
        return 0.5 * (alpha + drift) ** 2

    # 3. Integrate (Sum)
    # vmap over the sequence dimension
    energies = vmap(step_energy)(velocities, midpoints)

    return jnp.sum(energies)
