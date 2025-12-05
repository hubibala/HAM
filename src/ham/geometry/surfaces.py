import jax
import jax.numpy as jnp
from .manifold import Manifold

class Sphere(Manifold):
    """
    S^2 embedded in R^3.
    Constraints: |x| = radius.
    """
    def __init__(self, radius=1.0):
        self.radius = radius

    @property
    def ambient_dim(self) -> int:
        return 3
    
    @property
    def intrinsic_dim(self) -> int:
        return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Projects x onto the sphere surface."""
        return self.radius * x / jnp.linalg.norm(x)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Projects v onto the tangent plane at x."""
        n = x / jnp.linalg.norm(x)
        return v - jnp.dot(v, n) * n
    
    def random_sample(self, key: jax.Array, shape: tuple) -> jnp.ndarray:
        """Uniform sampling on S^2."""
        x = jax.random.normal(key, shape + (3,))
        return self.radius * x / jnp.linalg.norm(x, axis=-1, keepdims=True)