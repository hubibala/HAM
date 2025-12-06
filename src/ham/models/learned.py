import jax
import jax.numpy as jnp
import equinox as eqx
from ..geometry.metric import FinslerMetric
from ..geometry.manifold import Manifold
from ..nn.networks import VectorField, PSDMatrixField

class NeuralRanders(FinslerMetric, eqx.Module):
    """
    A Learnable Randers Metric parameterized by Neural Networks.
    """
    h_net: PSDMatrixField
    w_net: VectorField
    manifold: Manifold = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array, 
                 hidden_dim: int = 32, depth: int = 2,
                 use_fourier: bool = True):
        self.manifold = manifold
        self.epsilon = 1e-5
        k1, k2 = jax.random.split(key)
        dim = manifold.ambient_dim
        
        # Metric Tensor usually is low-frequency (terrain), so standard MLP is fine
        self.h_net = PSDMatrixField(dim, hidden_dim, depth, k1)
        
        # Wind often has high-frequency turbulence, so we enable Fourier Features
        # Scale=2.0 allows capturing wave numbers up to ~2*pi*2 ~ 12, covering R=4 easily.
        self.w_net = VectorField(dim, hidden_dim, depth, k2, 
                                 use_fourier=use_fourier, fourier_scale=3.0)

    def _get_zermelo_data(self, x: jnp.ndarray):
        H = self.h_net(x)
        W_raw = self.w_net(x)
        W_raw = self.manifold.to_tangent(x, W_raw)
        
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
        sqrt_D = jnp.sqrt(discriminant + 1e-12)
        return (sqrt_D - W_dot_v) / lam