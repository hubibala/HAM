import jax
import jax.numpy as jnp
from .base import Manifold

class Sphere(Manifold):
    """
    The n-dimensional Sphere S^n embedded in R^(n+1).
    Constraint: ||x|| = 1
    Tangent Space Condition: <x, v> = 0
    """
    def __init__(self, dim=2):
        # S^2 lives in R^3
        super().__init__(dim=dim, ambient_dim=dim+1)

    def projection(self, x: jnp.ndarray) -> jnp.ndarray:
        """Normalizes x to the unit sphere."""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        # Avoid division by zero with a safe epsilon
        return x / jnp.maximum(norm, 1e-12)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Projects vector v onto the tangent plane at x.
        Formula: v_tan = v - <x, v> * x  (Assuming x is normalized)
        """
        # Ensure x is on manifold first (safety check)
        x_on = self.projection(x)
        
        # Compute radial component <x, v>
        # keepdims=True ensures broadcasting works for batches (B, D)
        dot = jnp.sum(x_on * v, axis=-1, keepdims=True)
        
        # Remove the radial component
        return v - dot * x_on

    def random_uniform(self, key, shape: tuple) -> jnp.ndarray:
        """Samples from S^n via normalized Gaussian."""
        # If shape is (B,), we return (B, ambient_dim)
        final_shape = shape + (self.ambient_dim,)
        x = jax.random.normal(key, final_shape)
        return self.projection(x)