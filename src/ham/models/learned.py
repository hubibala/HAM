import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable

from ..geometry.metric import FinslerMetric
from ..geometry.manifold import Manifold
from ..nn.networks import VectorField, PSDMatrixField

class NeuralRanders(FinslerMetric, eqx.Module):
    """
    A Learnable Randers Metric parameterized by Neural Networks.
    Learns W(x) and h(x) to recover the underlying Finsler geometry.
    """
    h_net: PSDMatrixField
    w_net: VectorField
    manifold: Manifold = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array, 
                 hidden_dim: int = 32, depth: int = 2):
        self.manifold = manifold
        self.epsilon = 1e-5
        k1, k2 = jax.random.split(key)
        dim = manifold.ambient_dim
        self.h_net = PSDMatrixField(dim, hidden_dim, depth, k1)
        self.w_net = VectorField(dim, hidden_dim, depth, k2)

    def _get_zermelo_data(self, x: jnp.ndarray):
        # 1. Learned Terrain h(x)
        H = self.h_net(x)
        
        # 2. Learned Wind W(x)
        W_raw = self.w_net(x)
        
        # FIX: Project W onto the tangent space immediately.
        # This ensures we don't waste the "Zermelo Budget" (norm < 1) on
        # useless normal components that don't affect the physics.
        W_raw = self.manifold.to_tangent(x, W_raw)
        
        # 3. Zermelo Convexity Enforcement
        # |W|_h^2 = W^T H W
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(w_norm_sq + 1e-8)
        
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        
        W = W_raw * scale
        lam = 1.0 - (w_norm * scale)**2
        return H, W, lam

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        H, W, lam = self._get_zermelo_data(x)
        v_sq_h = jnp.dot(v, jnp.dot(H, v))
        W_dot_v = jnp.dot(v, jnp.dot(H, W))
        discriminant = lam * v_sq_h + W_dot_v**2
        return (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam