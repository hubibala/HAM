import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from ..geometry.metric import FinslerMetric
from ..geometry.manifold import Manifold
from ..geometry.zoo import Randers, Riemannian
from ..nn.networks import VectorField, PSDMatrixField
from ..utils.math import safe_norm

class NeuralRiemannian(Riemannian, eqx.Module):
    """
    A Learnable Riemannian Metric parameterized by Neural Networks.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array, 
                 hidden_dim: int = 32, depth: int = 2):
        self.dim = manifold.ambient_dim
        g_net = PSDMatrixField(self.dim, hidden_dim, depth, key)
        super().__init__(manifold, g_net)

class NeuralRanders(Randers, eqx.Module):
    """
    A Learnable Randers Metric parameterized by Neural Networks.
    Inherits from Randers to reuse the correct Zermelo cost logic.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array, 
                 hidden_dim: int = 32, depth: int = 2,
                 use_fourier: bool = True):
        k1, k2 = jax.random.split(key)
        self.dim = manifold.ambient_dim
        
        h_net = PSDMatrixField(self.dim, hidden_dim, depth, k1)
        w_net = VectorField(self.dim, hidden_dim, depth, k2, 
                            use_fourier=use_fourier, fourier_scale=3.0)
                            
        super().__init__(manifold, h_net, w_net, epsilon=1e-5)
    


class PullbackRanders(Randers):
    """
    A Randers Metric where H(z) is strictly defined by the Decoder's 
    geometry (Pullback Metric), and only the Wind W(z) is learned.
    """
    dim: int = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True, default=True)

    def __init__(self, manifold, decoder, key, hidden_dim=64, depth=3, use_fourier=False, fourier_scale=1.0, use_wind=True):
        self.dim = int(manifold.ambient_dim)
        self.use_wind = bool(use_wind)
        
        # We only need the VectorField for the Wind.
        w_net = VectorField(self.dim, hidden_dim, depth, key, 
                            use_fourier=use_fourier, fourier_scale=fourier_scale)
        
        # Proper eqx.Module for H(z)
        h_net = PullbackGNet(decoder=decoder, dim=self.dim)
        
        super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=self.use_wind)


class KernelWindField(eqx.Module):
    """
    Non-parametric nearest-neighbor kernel smoother for exact
    pseudo-velocities in latent space.
    """
    anchors_z: Any
    anchors_v: Any
    sigma: float = eqx.field(static=True)

    def __init__(self, anchors_z, anchors_v, sigma=0.5):
        self.anchors_z = anchors_z
        self.anchors_v = anchors_v
        self.sigma = float(sigma)

    def __call__(self, z: jax.Array) -> jax.Array:
        # z: (D,), anchors_z: (N, D)
        diff = self.anchors_z - z[None, :]          # (N, D)
        dists_sq = jnp.sum(diff ** 2, axis=-1)      # (N,)
        weights = jax.nn.softmax(-dists_sq / (2 * self.sigma ** 2))   # (N,)
        return jnp.dot(weights, self.anchors_v)      # (D,)

class DataDrivenPullbackRanders(Randers):
    """
    Instead of a parameterized neural network, uses a kernel smoother
    over the dataset's exact RNA velocities projected into the latent space.
    """
    dim: int = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True, default=True)

    def __init__(self, manifold, decoder, anchors_z, anchors_v, sigma=0.5, use_wind=True):
        self.dim = int(manifold.ambient_dim)
        self.use_wind = bool(use_wind)
        h_net = PullbackGNet(decoder=decoder, dim=self.dim)
        w_net = KernelWindField(anchors_z, anchors_v, sigma)
        super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=self.use_wind)


class PullbackGNet(eqx.Module):
    decoder: Any
    dim: Any = eqx.field(static=True)

    
    def __call__(self, z: jax.Array) -> jax.Array:
        J = jax.jacfwd(self.decoder)(z)
        H = jnp.dot(J.T, J)
        # Add a tiny diagonal for numerical stability and scale down to prevent
        # extreme causality squashing when eigenvalues are large.
        return (H + 1e-4 * jnp.eye(self.dim)) * 1e-4

class PullbackRiemannian(Riemannian):
    """
    A Riemannian Metric where G(z) is strictly defined by the Decoder's 
    geometry (Pullback Metric).
    """
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, decoder: eqx.Module, hidden_dim: int = 32):
        self.dim = manifold.ambient_dim
        g_net = PullbackGNet(decoder=decoder, dim=self.dim)
        super().__init__(manifold, g_net)