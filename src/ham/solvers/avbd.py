import jax
import jax.numpy as jnp
from typing import NamedTuple, List, Callable, Optional, Tuple
from functools import partial
import equinox as eqx

# Assuming FinslerMetric is defined in ham.geometry.metric
from ham.geometry.metric import FinslerMetric
from ham.utils.math import safe_norm

class Trajectory(NamedTuple):
    xs: jnp.ndarray
    vs: jnp.ndarray
    energy: jnp.ndarray
    constraint_violation: jnp.ndarray
    is_converged: jnp.ndarray
    final_gradient: jnp.ndarray

class SolverState(NamedTuple):
    path: jnp.ndarray          # Inner points (T-2, D)
    lambdas: jnp.ndarray       # Lagrange multipliers
    stiffness: jnp.ndarray     # Penalty parameters
    prev_path: jnp.ndarray     # For momentum
    step: int
    max_violation: float
    prev_energy: float
    curr_energy: float
    max_gradient: float

class AVBDSolver(eqx.Module):
    """
    Augmented Vertex Block Descent (AVBD) Geodesic Solver.

    A differentiable Boundary-Value Problem (BVP) solver that finds geodesics 
    by minimizing path energy over a discretized set of path points. Fully 
    JAX-compatible and autodiff-friendly.

    Attributes:
        step_size: Gradient descent learning rate for inner path points.
        iterations: Number of full relaxation passes per solve.
        tol: Tolerance for convergence (gradient norm).
        energy_tol: Tolerance for convergence (energy change).
    """
    step_size: float = 0.004  # effective rate is step_size/2 due to shared segment counting
    iterations: int = 150
    tol: float = 1e-4
    energy_tol: float = 1e-4

    def solve(self, metric: FinslerMetric, 
              p_start: jnp.ndarray, p_end: jnp.ndarray, 
              n_steps: int = 10,
              constraints: Optional[List[Callable]] = None,
              train_mode: bool = True,
              key: Optional[jax.Array] = None) -> Trajectory:
        
        if key is None:
            # Derive a data-dependent key so vmap'd calls get different
            # permutation orders instead of all sharing PRNGKey(42).
            # fold_in with a hash of the start/end points makes each
            # geodesic pair get a unique key under vmap.
            data_hash = jnp.sum(p_start * 1e4 + p_end * 1e2).astype(jnp.int32)
            key = jax.random.fold_in(jax.random.PRNGKey(42), data_hash)
        k1, k2 = jax.random.split(key)
        
        if constraints is None: constraints = []
        num_constraints = len(constraints)
        
        # 1. Linear Initialization
        t = jnp.linspace(0, 1, n_steps + 1)[:, None]
        # Linear interpolation in ambient space (or manifold if possible)
        # Note: If Hyperboloid, linear interp is "secant" lines, projected later
        linear_path = (1 - t) * p_start + t * p_end
        
        # CRITICAL FIX: Add noise to prevent zero-velocity segments
        # If start == end, velocity is 0, and gradients explode.
        noise = jax.random.normal(k1, shape=linear_path.shape) * 1e-4
        linear_path = linear_path + noise
        
        # Project initialization to manifold
        path_guess = jax.vmap(metric.manifold.project)(linear_path)
        init_inner = path_guess[1:-1]

        # Init Dual Variables
        n_inner = n_steps - 1
        init_lambdas = jnp.zeros((n_inner, num_constraints))
        init_stiffness = jnp.ones((n_inner, num_constraints))

        state = SolverState(
            path=init_inner,
            lambdas=init_lambdas,
            stiffness=init_stiffness,
            prev_path=init_inner,
            step=0,
            max_violation=1.0,
            prev_energy=jnp.inf,
            curr_energy=0.0,
            max_gradient=jnp.inf
        )

        # 2. Define the discrete Action (Energy) locally
        def local_action(x_prev, x, x_next):
            # Discrete Finsler Energy: F(x, v)^2
            v_in = metric.manifold.log_map(x_prev, x)
            v_out = metric.manifold.log_map(x, x_next)
            # We minimize the sum of energies of the two segments connected to x
            return metric.energy(x_prev, v_in) + metric.energy(x, v_out)

        # 3. Block Update Logic (Optimizing one vertex 'x')
        def update_vertex(full_path, idx, s):
            x_prev = full_path[idx]     # Neighbor Left
            x = full_path[idx+1]        # Current Node (Target)
            x_next = full_path[idx+2]   # Neighbor Right

            # A. Gradient Descent on Manifold
            def loss_fn(current_x):
                # Energy
                E = local_action(x_prev, current_x, x_next)
                
                # Constraints (Augmented Lagrangian)
                penalty = 0.0
                if num_constraints > 0:
                    c_val = jnp.stack([c(current_x) for c in constraints])
                    lam = s.lambdas[idx]
                    k = s.stiffness[idx]
                    penalty = jnp.sum(lam * c_val + 0.5 * k * (c_val**2))
                
                return E + penalty

            # Calculate Riemannian Gradient
            # grad_f is Euclidean gradient
            grad_f = jax.grad(loss_fn)(x)
            
            # Project to Tangent Space
            grad_tan = metric.manifold.to_tangent(x, grad_f)
            
            # Prevent exploding gradients by clipping the norm

            grad_norm = safe_norm(grad_tan)
            clip_value = 2.0
            grad_tan = jnp.where(grad_norm > clip_value, grad_tan * (clip_value / grad_norm), grad_tan)
            
            # Update Step (with Momentum if needed, simplified here for stability)
            # x_new = Retract(x, -lr * grad)
            step = -self.step_size * grad_tan
            x_new = metric.manifold.retract(x, step)
            
            return x_new, grad_norm

        # 4. The Loop Body
        def step_fn(s: SolverState, _):
            # Reconstruct full path for context
            full_path = jnp.concatenate([p_start[None, :], s.path, p_end[None, :]], axis=0)
            # Randomize update order (Stochastic Block Descent)
            step_key = jax.random.fold_in(k2, s.step) # Fold in step for reproducibility
            order = jax.random.permutation(step_key, jnp.arange(n_inner))

            # Re-index order to account for boundaries in the full path
            full_order = order + 1
            
            # Scan over vertices to update them
            def scan_body(curr_path_full, idx):
                new_node, grad_norm = update_vertex(curr_path_full, idx - 1, s)
                return curr_path_full.at[idx].set(new_node), grad_norm

            new_full, grad_norms = jax.lax.scan(scan_body, full_path, full_order)
            new_inner = new_full[1:-1]
            
            # Dual Updates (Constraints)
            # ... (Simplified: omitted for core geodesic regression unless needed) ...
            
            # Compute Energy for monitoring
            full_new = jnp.concatenate([p_start[None, :], new_inner, p_end[None, :]], axis=0)
            vels = jax.vmap(metric.manifold.log_map)(full_new[:-1], full_new[1:])
            total_E = jnp.sum(jax.vmap(metric.energy)(full_new[:-1], vels))
            
            return SolverState(
                path=new_inner,
                lambdas=s.lambdas,
                stiffness=s.stiffness,
                prev_path=s.path,
                step=s.step + 1,
                max_violation=0.0,
                prev_energy=s.curr_energy,
                curr_energy=total_E,
                max_gradient=jnp.max(grad_norms)
            ), None

        # 5. Execution Strategy
        if train_mode:
            # Fixed unrolling (differentiable!)
            final_state, _ = jax.lax.scan(step_fn, state, None, length=self.iterations)
        else:
            # While loop (convergence based, strictly for inference)
            def cond(s):
                not_converged = s.max_gradient > self.tol
                not_energy_converged = jnp.abs(s.curr_energy - s.prev_energy) > self.energy_tol
                under_limit = s.step < self.iterations
                return under_limit & (not_converged | not_energy_converged)

            final_state = jax.lax.while_loop(cond, lambda s: step_fn(s, None)[0], state)

        # 6. Output
        full_path = jnp.concatenate([p_start[None, :], final_state.path, p_end[None, :]], axis=0)
        velocities = jax.vmap(metric.manifold.log_map)(full_path[:-1], full_path[1:])
        
        return Trajectory(
            xs=full_path,
            vs=velocities,
            energy=final_state.curr_energy,
            constraint_violation=final_state.max_violation,
            is_converged=(final_state.max_gradient <= self.tol),
            final_gradient=final_state.max_gradient
        )