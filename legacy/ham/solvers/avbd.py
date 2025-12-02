import jax
import jax.numpy as jnp
from jax import jit, grad, lax, vmap
from jax.tree_util import register_dataclass
from typing import Callable, List, Any
from dataclasses import dataclass


@register_dataclass
@dataclass(frozen=True)
class SolverState:
    path: jnp.ndarray  # (T, D)
    lambdas: jnp.ndarray  # (T, C)
    stiffness: jnp.ndarray  # (T, C)
    step: int
    error: float


class AVBDSolver:
    """
    Robust AVBD Solver (Explicit Differentiation).

    This implementation accepts 'params' to allow gradients to flow back
    into the Metric Network during the 'Dream' phase.
    """

    def __init__(
        self, lr: float = 0.05, beta: float = 10.0, max_iters: int = 2000, tol: float = 1e-6
    ):
        self.lr = lr
        self.beta = beta
        self.max_iters = max_iters
        self.tol = tol

    def solve(
        self,
        params: Any,
        energy_fn_template: Callable[[Any, jnp.ndarray], float],
        constraints: List[Callable[[jnp.ndarray], float]],
        fixed_start: jnp.ndarray,
        fixed_end: jnp.ndarray,
        init_inner_path: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Args:
            params: Metric parameters (passed to energy_fn).
            energy_fn_template: f(params, full_path) -> scalar cost.
            constraints: List of c(x) -> scalar.
            fixed_start, fixed_end: Boundary conditions.
            init_inner_path: Initial guess (T, D).
        """
        num_points = init_inner_path.shape[0]
        num_constraints = len(constraints)

        init_state = SolverState(
            path=init_inner_path,
            lambdas=jnp.zeros((num_points, max(1, num_constraints))),
            stiffness=jnp.ones((num_points, max(1, num_constraints))),
            step=0,
            error=1e9,
        )

        def step_fn(i, current_state: SolverState):
            path, lam, k = current_state.path, current_state.lambdas, current_state.stiffness

            # 1. Primal Update (Gradient Descent on Lagrangian)
            def aug_L(p):
                full_p = jnp.concatenate([fixed_start[None, :], p, fixed_end[None, :]])
                # Note: We pass 'params' here
                E = energy_fn_template(params, full_p)

                if num_constraints > 0:

                    def c_cost(pt, l_pt, k_pt):
                        cost = 0.0
                        for idx, c_fn in enumerate(constraints):
                            val = c_fn(pt)
                            cost += l_pt[idx] * val + 0.5 * k_pt[idx] * (val**2)
                        return cost

                    C_term = jnp.sum(vmap(c_cost)(p, lam, k))
                else:
                    C_term = 0.0
                return E + C_term

            grads = grad(aug_L)(path)
            new_path = path - self.lr * grads

            # 2. Dual Update (Constraint Hardening)
            if num_constraints > 0:

                def get_violations(pt):
                    return jnp.stack([c(pt) for c in constraints])

                C_vals = vmap(get_violations)(new_path)
                new_lam = lam + k * C_vals
                new_k = k + self.beta * jnp.abs(C_vals)
                max_violation = jnp.max(jnp.abs(C_vals))
            else:
                new_lam, new_k, max_violation = lam, k, 0.0

            return SolverState(new_path, new_lam, new_k, i, max_violation)

        final_state = lax.fori_loop(0, self.max_iters, step_fn, init_state)
        return jnp.concatenate([fixed_start[None, :], final_state.path, fixed_end[None, :]])
