import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, lax, vmap  # <--- Added vmap here
from typing import Callable
from ..manifolds.base import Manifold

class ProjectedGradientSolver:
    """
    JAX-native solver for geodesic problems on manifolds.
    Minimizes an energy function while enforcing manifold constraints via projection.
    """
    def __init__(self, manifold: Manifold, lr: float = 0.01, max_iters: int = 500):
        self.manifold = manifold
        self.lr = lr
        self.max_iters = max_iters

    def solve(self, energy_fn: Callable, fixed_start: jnp.ndarray, fixed_end: jnp.ndarray, init_inner_path: jnp.ndarray):
        """
        Solves for the optimal path between two points.
        
        Args:
            energy_fn: Function f(inner_points) -> scalar cost.
            fixed_start: (Dim,)
            fixed_end: (Dim,)
            init_inner_path: (Steps-2, Dim) Initial guess for intermediate points.
        """
        
        # We wrap the energy function to handle the fixed endpoints internally
        def objective(inner_pts):
            # Reconstruct full path [Start, ..., End]
            full_path = jnp.concatenate([
                fixed_start[None, :],
                inner_pts,
                fixed_end[None, :]
            ])
            return energy_fn(full_path)

        # The Update Step (Compiled)
        def update_step(i, pts):
            grads = jax.grad(objective)(pts)
            new_pts = pts - self.lr * grads
            
            # CRITICAL: Project back to Manifold
            # We use vmap to apply the projection to every point in the path
            return vmap(self.manifold.projection)(new_pts)

        # Run Loop
        final_inner = lax.fori_loop(0, self.max_iters, update_step, init_inner_path)
        
        # Return full path
        return jnp.concatenate([
            fixed_start[None, :],
            final_inner,
            fixed_end[None, :]
        ])