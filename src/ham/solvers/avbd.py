import jax
import jax.numpy as jnp
from jax import jit, grad, lax, vmap
from jax.tree_util import register_dataclass
from typing import Callable, List
from dataclasses import dataclass


@register_dataclass
@dataclass(frozen=True)
class SolverState:
    path: jnp.ndarray  # (T, D)
    lambdas: jnp.ndarray  # (T, C)
    stiffness: jnp.ndarray  # (T, C)


class AVBDSolver:
    """
    Augmented Vertex Block Descent (AVBD) Solver.
    Solves constrained optimization problems using a primal-dual approach
    with adaptive stiffness (Progressive Hardening).
    """

    def __init__(self, lr: float = 0.01, beta: float = 10.0, max_iters: int = 1000):
        self.lr = lr
        self.beta = beta  # Stiffness ramping factor
        self.max_iters = max_iters

    def solve(
        self,
        energy_fn: Callable,
        constraints: List[Callable],
        fixed_start: jnp.ndarray,
        fixed_end: jnp.ndarray,
        init_inner_path: jnp.ndarray,
    ):
        """
        Args:
            energy_fn: f(path) -> scalar cost
            constraints: List of functions c(x) -> scalar. (Constraint is c(x)=0)
            fixed_start, fixed_end: Boundary conditions.
            init_inner_path: Initial guess.
        """
        num_points = init_inner_path.shape[0]
        num_constraints = len(constraints)

        # Initialize Dual Variables (Shape (T, 0) if no constraints, which is valid in JAX)
        init_lambdas = jnp.zeros((num_points, num_constraints))
        init_stiffness = jnp.ones((num_points, num_constraints))

        state = SolverState(init_inner_path, init_lambdas, init_stiffness)

        # ---------------------------------------------------------
        # 1. The Augmented Lagrangian (Energy + Constraints)
        # ---------------------------------------------------------
        def augmented_lagrangian(inner_path, lambdas, stiffness):
            # Reconstruct full path for physics energy
            full_path = jnp.concatenate([fixed_start[None, :], inner_path, fixed_end[None, :]])

            # A. Physics Energy
            E_phys = energy_fn(full_path)

            # B. Constraint Energy
            if num_constraints > 0:

                def eval_constraints_at_point(pt, l_pt, k_pt):
                    cost = 0.0
                    for i, c_fn in enumerate(constraints):
                        val = c_fn(pt)
                        # Lagrangian: lambda*C + 0.5*k*C^2
                        cost += l_pt[i] * val + 0.5 * k_pt[i] * (val**2)
                    return cost

                E_const = jnp.sum(vmap(eval_constraints_at_point)(inner_path, lambdas, stiffness))
            else:
                E_const = 0.0

            return E_phys + E_const

        # ---------------------------------------------------------
        # 2. The Update Step (Primal-Dual)
        # ---------------------------------------------------------
        def step(i, current_state: SolverState):
            path, lam, k = current_state.path, current_state.lambdas, current_state.stiffness

            # --- PRIMAL STEP (Gradient Descent) ---
            # This works regardless of constraints because augmented_lagrangian handles the logic
            grads = grad(lambda p: augmented_lagrangian(p, lam, k))(path)
            new_path = path - self.lr * grads

            # --- DUAL STEP (Update Multipliers) ---
            if num_constraints > 0:

                def get_violations(pt):
                    return jnp.stack([c(pt) for c in constraints])

                # C_vals shape: (T, NumConstraints)
                C_vals = vmap(get_violations)(new_path)

                # Update Lambda: lam = lam + k * C
                new_lam = lam + k * C_vals

                # Update Stiffness: k = k + beta * |C|
                new_k = k + self.beta * jnp.abs(C_vals)
            else:
                # No constraints to update, keep state
                new_lam = lam
                new_k = k

            return SolverState(new_path, new_lam, new_k)

        # ---------------------------------------------------------
        # 3. Run Loop
        # ---------------------------------------------------------
        final_state = lax.fori_loop(0, self.max_iters, step, state)

        return jnp.concatenate([fixed_start[None, :], final_state.path, fixed_end[None, :]])
