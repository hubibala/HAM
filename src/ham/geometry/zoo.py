import jax
import jax.numpy as jnp
from typing import Callable

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold

class Euclidean(FinslerMetric):
    """
    Standard Euclidean metric: F(x, v) = |v|.
    Curvature is zero.
    """
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(v)

class Riemannian(FinslerMetric):
    """
    General Riemannian metric: F(x, v) = sqrt( v^T G(x) v ).
    Defined by a position-dependent positive-definite matrix G(x).
    """
    def __init__(self, manifold: Manifold, g_net: Callable[[jnp.ndarray], jnp.ndarray]):
        """
        Args:
            manifold: The underlying domain.
            g_net: A function (or Neural Network) mapping x -> (D, D) matrix.
                   Must return a positive-definite matrix.
        """
        super().__init__(manifold)
        self.g_net = g_net

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        G_x = self.g_net(x)
        # Numerical stability: Ensure symmetry
        G_x = 0.5 * (G_x + G_x.T)
        
        # Compute norm
        quad = jnp.dot(v, jnp.dot(G_x, v))
        return jnp.sqrt(jnp.maximum(quad, 1e-6))

class Randers(FinslerMetric):
    """
    Randers Metric defined via Zermelo Navigation.
    F(x, v) = sqrt( |v|_h^2 + <W, v>_h^2 ) - <W, v>_h / lambda
    
    where:
    - h(x): Riemannian metric (The 'Sea')
    - W(x): Wind field (The 'Current')
    - lambda = 1 - |W|_h^2
    
    Constraint: |W|_h < 1 (Wind must be weaker than the boat's max speed).
    
    Reference: MATH_SPEC.md, Section 5
    """
    def __init__(self, 
                 manifold: Manifold, 
                 h_net: Callable[[jnp.ndarray], jnp.ndarray],
                 w_net: Callable[[jnp.ndarray], jnp.ndarray],
                 epsilon: float = 1e-5):
        super().__init__(manifold)
        self.h_net = h_net
        self.w_net = w_net
        self.epsilon = epsilon

    def _get_zermelo_data(self, x: jnp.ndarray):
        """
        Computes h(x) and W(x), enforcing the convexity constraint |W|_h < 1.
        Implementation harvested from legacy 'RandersFactory'.
        """
        # 1. Get Base Riemannian Metric h(x)
        H = self.h_net(x)
        H = 0.5 * (H + H.T) # Enforce symmetry
        
        # 2. Get Raw Wind W_raw(x)
        W_raw = self.w_net(x)
        
        # 3. Calculate Wind Norm in metric h
        # |W|_h^2 = W^T H W
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(w_norm_sq + 1e-8)
        
        # 4. Stabilize Wind (Convexity Constraint)
        # We need |W|_h < 1. We use tanh to squash the norm softy.
        # factor = tanh(w_norm) / w_norm
        # If w_norm is very small, factor -> 1.
        # We perform a safe division (or use logic where we just scale if > 1-eps).
        # The legacy code used: scaled_W = W * (tanh(norm) / norm) * 0.95
        
        # Strict enforcement: max_norm = 1.0 - epsilon
        # scale = (1 - epsilon) * tanh(w_norm) / w_norm
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        
        W = W_raw * scale
        
        # Re-calculate correct norm for lambda
        # This is safe because we just scaled W
        w_final_norm_sq = w_norm_sq * (scale**2)
        lam = 1.0 - w_final_norm_sq
        
        return H, W, lam

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        H, W, lam = self._get_zermelo_data(x)
        
        # Compute terms
        # v_norm_h^2 = v^T H v
        v_sq_h = jnp.dot(v, jnp.dot(H, v))
        
        # <W, v>_h = v^T H W
        W_dot_v = jnp.dot(v, jnp.dot(H, W))
        
        # Discriminant: lambda * |v|^2 + <W,v>^2
        discriminant = lam * v_sq_h + W_dot_v**2
        
        # F = (sqrt(discriminant) - <W,v>) / lambda
        # Note: We use minus sign for W term per MATH_SPEC.md Section 5
        # "Headwind (opposing W) increases cost."
        cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam
        
        return cost