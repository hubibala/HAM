import jax
import jax.numpy as jnp
from typing import NamedTuple

from ham.geometry.metric import FinslerMetric

class Trajectory(NamedTuple):
    xs: jnp.ndarray
    vs: jnp.ndarray
    energy: jnp.ndarray

class AVBDSolver:
    """
    Augmented Vertex Block Descent (AVBD) Solver.
    
    A robust variational integrator that solves the Boundary Value Problem (BVP)
    by minimizing the discrete Action functional:
    
    S[x] = Sum_i ( Energy(x_i, v_i) ) + Constraints
    
    It handles:
    1. Manifold constraints (via projection).
    2. Boundary conditions (fixed start/end).
    3. Metric geometry (via FinslerMetric).
    
    Reference: ARCH_SPEC.md, Section 4.2
    """
    
    def __init__(self, step_size: float = 0.1, max_iter: int = 100, tol: float = 1e-4):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, metric: FinslerMetric, 
              p_start: jnp.ndarray, p_end: jnp.ndarray, 
              n_steps: int = 20) -> Trajectory:
        """
        Finds the energy-minimizing geodesic between p_start and p_end.
        """
        # 1. Initialize path (Linear interpolation)
        # Note: This linear init might be off-manifold, but the loop fixes it.
        t = jnp.linspace(0, 1, n_steps + 1)
        # Reshape for broadcasting: (N, 1) * (D,) + (N, 1) * (D,)
        # p_start, p_end are (D,)
        t = t[:, None] 
        init_path = (1 - t) * p_start + t * p_end
        
        # 2. Optimization Loop (Projected Gradient Descent on the Path)
        # We optimize inner points x_1 ... x_{N-1}
        
        def action_loss(path_inner):
            # Reconstruct full path: [start, inner, end]
            full_path = jnp.concatenate([p_start[None, :], path_inner, p_end[None, :]], axis=0)
            
            # Compute velocities (finite difference)
            # v_i ~ (x_{i+1} - x_i)
            # Note: For time-fixed geodesics, Energy is proportional to Length^2.
            # E = 0.5 * F(x, v)^2
            
            xs = full_path[:-1]
            vs = full_path[1:] - full_path[:-1]
            
            # Vectorize energy computation
            # We assume metric.energy handles single points, so we vmap it.
            # metric.energy(x, v) -> scalar
            energies = jax.vmap(metric.energy)(xs, vs)
            
            return jnp.sum(energies)

        grad_fn = jax.value_and_grad(action_loss)
        
        # Initial guess for inner points
        path_inner = init_path[1:-1]
        
        # JIT compile the update step for speed
        @jax.jit
        def update_step(i, state):
            p_inner, _ = state
            loss, grads = grad_fn(p_inner)
            
            # Gradient Descent
            p_new = p_inner - self.step_size * grads
            
            # Project onto manifold
            # We map the projection over the batch of points
            p_proj = jax.vmap(metric.manifold.project)(p_new)
            
            return p_proj, loss

        # Run optimization
        # We use lax.fori_loop for efficiency
        final_inner, final_loss = jax.lax.fori_loop(0, self.max_iter, update_step, (path_inner, 0.0))
        
        # 3. Assemble Result
        full_path = jnp.concatenate([p_start[None, :], final_inner, p_end[None, :]], axis=0)
        velocities = full_path[1:] - full_path[:-1]
        
        return Trajectory(xs=full_path, vs=velocities, energy=final_loss)