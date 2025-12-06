import jax
import jax.numpy as jnp
from .manifold import Manifold
from ..utils.math import safe_norm

class Sphere(Manifold):
    """S^2 embedded in R^3."""
    def __init__(self, radius=1.0):
        self.radius = radius

    @property
    def ambient_dim(self) -> int: return 3
    @property
    def intrinsic_dim(self) -> int: return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.radius * x / safe_norm(x)

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # P(x) = r * x / |x| => dP = r/|x| * (I - n n^T)
        n = x / safe_norm(x)
        return v - jnp.dot(v, n) * n
    
    def random_sample(self, key: jax.Array, shape: tuple) -> jnp.ndarray:
        """Uniform sampling on S^2."""
        x = jax.random.normal(key, shape + (3,))
        return self.project(x)

class Torus(Manifold):
    """
    Torus T^2 embedded in R^3.
    Distance to central ring = minor_r.
    """
    def __init__(self, major_R=2.0, minor_r=0.5):
        self.R = major_R
        self.r = minor_r

    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 2

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        # 1. Project to the "spine" circle on XY plane
        xy_dist = safe_norm(x[:2])
        # Scale factor to move x to radius R
        scale = self.R / jnp.maximum(xy_dist, 1e-12)
        spine_pt = jnp.array([x[0] * scale, x[1] * scale, 0.0])
        
        # 2. Project from spine to surface (distance r)
        vec = x - spine_pt
        dist = safe_norm(vec)
        surf_scale = self.r / jnp.maximum(dist, 1e-12)
        return spine_pt + vec * surf_scale

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        # Automatic tangent space via differentiation of projection
        _, v_proj = jax.jvp(self.project, (x,), (v,))
        return v_proj

    def random_sample(self, key: jax.Array, shape: tuple) -> jnp.ndarray:
        """
        Samples points on the torus via parameterization.
        Note: This is not strictly area-uniform due to the Jacobian determinant (R + r cos theta),
        but it produces valid points on the manifold for testing.
        """
        k1, k2 = jax.random.split(key)
        # Angles u (tube) and v (spine)
        u = jax.random.uniform(k1, shape) * 2 * jnp.pi
        v = jax.random.uniform(k2, shape) * 2 * jnp.pi
        
        # Torus parameterization
        # x = (R + r cos u) cos v
        # y = (R + r cos u) sin v
        # z = r sin u
        
        tube_r = self.R + self.r * jnp.cos(u)
        x = tube_r * jnp.cos(v)
        y = tube_r * jnp.sin(v)
        z = self.r * jnp.sin(u)
        
        return jnp.stack([x, y, z], axis=-1)

class Paraboloid(Manifold):
    """
    Open surface defined by z = x^2 + y^2.
    """
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 2
    
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        # Approximate projection: Vertical mapping.
        # Note: This is not the geometric closest point, but it guarantees 
        # the point lies on the manifold, which satisfies the constraint C(x)=0.
        return jnp.array([x[0], x[1], x[0]**2 + x[1]**2])

    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        _, v_proj = jax.jvp(self.project, (x,), (v,))
        return v_proj

    def random_sample(self, key: jax.Array, shape: tuple) -> jnp.ndarray:
        """Samples x,y from Normal(0,1) and computes z."""
        xy = jax.random.normal(key, shape + (2,))
        z = jnp.sum(xy**2, axis=-1, keepdims=True)
        return jnp.concatenate([xy, z], axis=-1)