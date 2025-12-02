import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

from ham.geometry.manifold import Manifold

class FinslerMetric(ABC):
    """
    The abstract base class for all Finsler metrics.
    
    This class implements the 'Metric-First' design pattern. Subclasses need only
    define `metric_fn(x, v)`. The energy, geodesic spray (equations of motion),
    and inner products are automatically derived via JAX autodiff.
    
    Reference: ARCH_SPEC.md, Section 2.2
    Reference: MATH_SPEC.md, Section 2 (Dynamics)
    """

    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    @abstractmethod
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        The fundamental Finsler cost function F(x, v).
        
        Args:
            x: Point on manifold (shape: D)
            v: Tangent vector at x (shape: D)
            
        Returns:
            Scalar cost (shape: ()).
            
        Constraint:
            Must be 1-homogeneous in v: F(x, k*v) == k * F(x, v)
        """
        pass

    def energy(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Lagrangian energy L = 0.5 * F(x, v)^2.
        
        This is the root of the computational graph for all geometric objects.
        """
        return 0.5 * self.metric_fn(x, v)**2

    def inner_product(self, x: jnp.ndarray, v: jnp.ndarray, 
                      w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Finsler inner product <w1, w2>_v.
        
        In Finsler geometry, the inner product depends on the direction 'v'.
        It is defined as the Hessian of the Energy function.
        
        g_ij(x, v) = d^2/dv_i dv_j (Energy)
        """
        # Hessian of E w.r.t v, evaluated at (x, v)
        g_fn = jax.hessian(self.energy, argnums=1)
        g_x_v = g_fn(x, v)
        
        return jnp.dot(w1, jnp.dot(g_x_v, w2))

    def spray(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Geodesic Spray coefficients G^i(x, v).
        
        The geodesic equation is: x'' + 2G(x, x') = 0.
        
        Implementation:
            Solves the linear system derived from Euler-Lagrange:
            Hess_v(E) * (x'') = Grad_x(E) - Jac_x(Grad_v(E)) * v
            Then G = -0.5 * x''
            
            This avoids explicit matrix inversion of g_ij.
            
        Reference: MATH_SPEC.md, Section 2.2
        """
        # 1. Compute gradients and Hessians needed for Euler-Lagrange
        grad_v_fn = jax.grad(self.energy, argnums=1)
        # Note: grad_v value is not needed directly, only the operator for JVP
        
        # grad_x_E (dL/dx)
        grad_x = jax.grad(self.energy, argnums=0)(x, v)
        
        # mixed_term = (d/dx (dL/dv)) * v
        # We calculate: \partial_x ( \partial_v E ) \cdot v
        # using JVP on the function f(x) = grad_v(x, v_fixed)
        def d_dv_fixed_v(pos):
            return grad_v_fn(pos, v)
            
        # jvp returns (primals_out, tangents_out). We want tangents_out.
        _, mixed_term = jax.jvp(d_dv_fixed_v, (x,), (v,))
        
        # RHS = Grad_x(E) - Mixed_Term
        # This represents the 'force' term in the linear system H*acc = Force
        rhs = grad_x - mixed_term
        
        # LHS Matrix = Hess_v(E) = g_ij
        hess_v = jax.hessian(self.energy, argnums=1)(x, v)
        
        # 2. Solve the linear system: hess_v * (acc) = rhs
        # acc represents \ddot{x}
        acc = jnp.linalg.solve(hess_v, rhs)
        
        # 3. Convert to Spray G
        # Definition: \ddot{x} + 2G = 0  =>  G = -0.5 * \ddot{x}
        return -0.5 * acc

    def geod_acceleration(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Returns x'' = -2G(x, v).
        Convenience wrapper for ODE solvers.
        """
        return -2.0 * self.spray(x, v)