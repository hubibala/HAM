import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Tuple

class Manifold(ABC):
    """
    Abstract base class for HAM manifolds.
    Defines how points and vectors are constrained to the space.
    """
    
    def __init__(self, dim: int, ambient_dim: int):
        self.dim = dim
        self.ambient_dim = ambient_dim

    @abstractmethod
    def projection(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Project a point from ambient space R^N onto the manifold M.
        Ensures x is 'on' the surface.
        """
        pass

    @abstractmethod
    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Project a vector v onto the tangent space T_x M.
        Ensures v is 'tangent' to the surface at x.
        """
        pass

    @abstractmethod
    def random_uniform(self, key, shape: tuple) -> jnp.ndarray:
        """Sample points uniformly on the manifold."""
        pass