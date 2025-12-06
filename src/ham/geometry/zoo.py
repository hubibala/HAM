
import jax
import jax.numpy as jnp
from typing import Callable, Optional

from ham.geometry.metric import FinslerMetric
from ham.geometry.manifold import Manifold
from ham.geometry.mesh import TriangularMesh
from ham.utils.math import safe_norm

class Euclidean(FinslerMetric):
    """Standard Euclidean metric: F(x, v) = |v|."""
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        return safe_norm(v)

class Riemannian(FinslerMetric):
    """General Riemannian metric: F(x, v) = sqrt( v^T G(x) v )."""
    def __init__(self, manifold: Manifold, g_net: Callable[[jnp.ndarray], jnp.ndarray]):
        super().__init__(manifold)
        self.g_net = g_net

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        G_x = self.g_net(x)
        G_x = 0.5 * (G_x + G_x.T)
        quad = jnp.dot(v, jnp.dot(G_x, v))
        return jnp.sqrt(jnp.maximum(quad, 1e-12))

class Randers(FinslerMetric):
    """
    Rigorous Randers Metric (Zermelo Navigation).
    F(x, v) = ( sqrt( lambda |v|_h^2 + <W, v>_h^2 ) - <W, v>_h ) / lambda
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
        H = self.h_net(x)
        H = 0.5 * (H + H.T) 
        
        W_raw = self.w_net(x)
        
        # Norm in metric H
        w_norm_sq = jnp.dot(W_raw, jnp.dot(H, W_raw))
        w_norm = jnp.sqrt(w_norm_sq + 1e-8)
        
        # Enforce convexity: ||W|| < 1
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        W = W_raw * scale
        
        # Lambda = 1 - ||W||^2
        w_final_sq = w_norm_sq * (scale**2)
        lam = 1.0 - w_final_sq
        
        return H, W, lam

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        H, W, lam = self._get_zermelo_data(x)
        
        v_sq_h = jnp.dot(v, jnp.dot(H, v))
        W_dot_v = jnp.dot(v, jnp.dot(H, W))
        
        discriminant = lam * v_sq_h + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-12)) - W_dot_v) / lam
        return cost

class PiecewiseConstantFinsler(FinslerMetric):
    """
    Metric on a mesh.
    If smooth=False: Cost is constant per face. Gradient is 0 (Optimizer won't work!).
    If smooth=True:  Cost is interpolated from vertices. Gradient exists.
    """
    def __init__(self, mesh: TriangularMesh, face_costs: jnp.ndarray, smooth: bool = True):
        super().__init__(mesh)
        if not isinstance(mesh, TriangularMesh):
            raise ValueError("Requires TriangularMesh.")
        self.face_costs = face_costs
        self.smooth = smooth
        
        if self.smooth:
            # Precompute vertex costs by averaging adjacent faces
            N_v = mesh.vertices.shape[0]
            v_costs = jnp.zeros(N_v)
            counts = jnp.zeros(N_v)
            
            # Expand face_costs for each vertex of the face
            flat_indices = mesh.faces.flatten()
            repeated_costs = jnp.repeat(face_costs, 3)
            
            v_costs = v_costs.at[flat_indices].add(repeated_costs)
            counts = counts.at[flat_indices].add(1.0)
            
            self.vertex_costs = v_costs / jnp.maximum(counts, 1.0)
        else:
            self.vertex_costs = None

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        face_idx = self.manifold.get_face_index(x)
        
        if self.smooth:
            # Interpolate cost using barycentric coordinates
            tri = self.manifold.triangles[face_idx]
            a, b, c = tri[0], tri[1], tri[2]
            
            ab, ac, ap = b - a, c - a, x - a
            d1, d2 = jnp.dot(ab, ap), jnp.dot(ac, ap)
            d3, d4, d5 = jnp.dot(ab, ab), jnp.dot(ab, ac), jnp.dot(ac, ac)
            
            det = jnp.maximum(d3 * d5 - d4 * d4, 1e-12)
            s = (d5 * d1 - d4 * d2) / det
            t = (d3 * d2 - d4 * d1) / det
            w = 1.0 - s - t
            
            face_verts = self.manifold.faces[face_idx]
            c_a = self.vertex_costs[face_verts[0]]
            c_b = self.vertex_costs[face_verts[1]]
            c_c = self.vertex_costs[face_verts[2]]
            
            cost = w * c_a + s * c_b + t * c_c
            cost = jnp.maximum(cost, 1e-2)
            
        else:
            cost = self.face_costs[face_idx]
            
        return cost * safe_norm(v)

class DiscreteRanders(FinslerMetric):
    """
    Anisotropic Mesh Metric (Wind per face).
    F(x, v) = Zermelo(v, W_face)
    """
    def __init__(self, mesh: TriangularMesh, face_winds: jnp.ndarray, epsilon: float = 1e-5):
        super().__init__(mesh)
        self.face_winds = face_winds
        self.epsilon = epsilon
        if not isinstance(mesh, TriangularMesh): raise ValueError("Requires TriangularMesh.")

    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        face_idx = self.manifold.get_face_index(x)
        W_raw = self.face_winds[face_idx]
        
        # Stabilize Wind
        w_norm = jnp.linalg.norm(W_raw)
        scale = (1.0 - self.epsilon) * jnp.tanh(w_norm) / (w_norm + 1e-8)
        W = W_raw * scale
        lam = 1.0 - (w_norm * scale)**2
        
        v_sq = jnp.dot(v, v)
        W_dot_v = jnp.dot(W, v)
        
        discriminant = lam * v_sq + W_dot_v**2
        cost = (jnp.sqrt(jnp.maximum(discriminant, 1e-8)) - W_dot_v) / lam
        return cost