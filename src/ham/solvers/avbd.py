import jax
import jax.numpy as jnp
from typing import NamedTuple
from ..geometry.metric import FinslerMetric

class Trajectory(NamedTuple):
    xs: jnp.ndarray
    vs: jnp.ndarray
    energy: jnp.ndarray

class AVBDSolver:
    """
    Augmented Vertex Block Descent (AVBD) Solver.
    Minimizes Action S[x] = Sum_i ( Energy(x_i, v_i) ) subject to constraints.
    """
    def __init__(self, step_size: float = 0.1, max_iter: int = 100, tol: float = 1e-4):
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, metric: FinslerMetric, 
              p_start: jnp.ndarray, p_end: jnp.ndarray, 
              n_steps: int = 20) -> Trajectory:
        
        # 1. Linear Initialization with Perturbation
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        linear_path = (1 - t) * p_start + t * p_end
        
        # FIX: Reduced noise scale from 0.05 to 1e-3.
        # This is sufficient to avoid the exact 0.0 singularity (Antipodal issue)
        # without introducing significant error in flat Euclidean cases.
        key = jax.random.PRNGKey(1337)
        noise = jax.random.normal(key, shape=linear_path.shape) * 1e-3
        
        # Project initialization onto manifold
        init_path = jax.vmap(metric.manifold.project)(linear_path + noise)
        
        # 2. Optimization Loop
        def action_loss(path_inner):
            full_path = jnp.concatenate([p_start[None, :], path_inner, p_end[None, :]], axis=0)
            xs = full_path[:-1]
            vs = full_path[1:] - full_path[:-1]
            # Vectorized energy computation
            energies = jax.vmap(metric.energy)(xs, vs)
            return jnp.sum(energies)

        grad_fn = jax.value_and_grad(action_loss)
        path_inner = init_path[1:-1]
        
        @jax.jit
        def update_step(i, state):
            p_inner, _ = state
            loss, grads = grad_fn(p_inner)
            # Gradient Descent
            p_new = p_inner - self.step_size * grads
            # Project constraint
            p_proj = jax.vmap(metric.manifold.project)(p_new)
            return p_proj, loss

        final_inner, final_loss = jax.lax.fori_loop(0, self.max_iter, update_step, (path_inner, 0.0))
        
        # 3. Assemble Result
        full_path = jnp.concatenate([p_start[None, :], final_inner, p_end[None, :]], axis=0)
        velocities = full_path[1:] - full_path[:-1]
        
        return Trajectory(xs=full_path, vs=velocities, energy=final_loss)