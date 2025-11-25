import jax
import jax.numpy as jnp
from dataclasses import dataclass
from jax.tree_util import register_dataclass
from ..manifolds.base import Manifold

@register_dataclass
@dataclass(frozen=True)
class RandersMetric:
    """
    A verified, valid Randers metric at specific points.
    F(v) = sqrt(v.T a v) + beta.T v
    """
    a: jnp.ndarray      # Metric Tensor (B, D, D)
    beta: jnp.ndarray   # The 1-form 'drift' (B, D)
    L: jnp.ndarray      # Cholesky of 'a' (B, D, D), cached for solver

class RandersFactory:
    """
    The 'Constructor' layer for HAM.
    Takes raw Neural Network outputs and produces a physically valid RandersMetric.
    """
    def __init__(self, manifold: Manifold, epsilon=0.05):
        self.manifold = manifold
        self.epsilon = epsilon

    def forward(self, x: jnp.ndarray, raw_L: jnp.ndarray, raw_W: jnp.ndarray) -> RandersMetric:
        """
        Args:
            x: Points on manifold (Batch, AmbientDim)
            raw_L: Raw shape parameters (Batch, AmbientDim) (Diagonal approximation)
            raw_W: Raw wind parameters (Batch, AmbientDim)
        """
        batch_shape = x.shape[:-1]
        dim = self.manifold.ambient_dim
        
        # --- 1. Construct Shape (L) ---
        diag_vals = jax.nn.softplus(raw_L) + 0.1 
        
        if raw_L.ndim == len(batch_shape) + 1:
            L = jnp.vectorize(jnp.diag, signature='(n)->(n,n)')(diag_vals)
        else:
            L = raw_L

        # --- 2. Force Tangency ---
        W_tangent = self.manifold.to_tangent(x, raw_W)
        
        # --- 3. Enforce Convexity ---
        LW = jnp.squeeze(jnp.matmul(jnp.swapaxes(L, -1, -2), W_tangent[..., None]), -1)
        
        # CRITICAL FIX: SAFE NORM
        # Standard jnp.linalg.norm produces NaN gradients at 0.
        # We implement sqrt(sum(x^2) + eps) manually.
        norm_sq = jnp.sum(LW**2, axis=-1, keepdims=True)
        norm_LW = jnp.sqrt(norm_sq + 1e-12) # Safe gradient at 0
        
        # Scale factor
        # Note: We removed the 1e-9 from denominator because norm_LW is now safe
        scale_factor = jnp.tanh(norm_LW) * (1.0 - self.epsilon) / norm_LW
        
        # Calculate beta
        beta = W_tangent * scale_factor
        
        # Re-ensure tangency (safety)
        beta = self.manifold.to_tangent(x, beta)
        
        # Compute 'a' = L @ L.T
        a = jnp.einsum('...ij,...kj->...ik', L, L)
        
        return RandersMetric(a=a, beta=beta, L=L)