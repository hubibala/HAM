import jax
import jax.numpy as jnp
from jax import jit, grad, jacfwd, hessian
from functools import partial

# --- 1. FINSLER GEOMETRY KERNEL ---


def finsler_energy(theta, x, v, metric_fn):
    """The Lagrangian L = 0.5 * F(x, v)^2."""
    g, beta = metric_fn(theta, x)

    # Robust Randers Norm
    vgv = jnp.dot(v, jnp.dot(g, v))
    alpha = jnp.sqrt(jnp.maximum(vgv, 1e-9))
    drift = jnp.dot(beta, v)

    F = alpha + drift
    return 0.5 * F**2


def compute_spray(theta, x, v, metric_fn):
    """Computes Geodesic Spray G^i via Auto-Diff of Euler-Lagrange."""
    L = lambda x_, v_: finsler_energy(theta, x_, v_, metric_fn)

    dL_dx = grad(L, argnums=0)(x, v)

    # Hessian w.r.t v
    H = hessian(L, argnums=1)(x, v)
    H_inv = jnp.linalg.inv(H + 1e-6 * jnp.eye(x.shape[0]))

    # Mixed Partial d^2L / dv dx
    M = jacfwd(grad(L, argnums=1), argnums=0)(x, v)

    # Spray Equation: H * a + M * v - dL_dx = 0
    rhs = dL_dx - jnp.dot(M, v)
    accel = jnp.dot(H_inv, rhs)

    return -0.5 * accel


def compute_berwald_connection(theta, x, v, metric_fn):
    """Computes Berwald Connection coefficients via Jacobian of Spray."""
    # Gamma = dG / dv
    return jacfwd(lambda v_: compute_spray(theta, x, v_, metric_fn))(v)


# --- 2. TRANSPORT INTEGRATOR ---


def transport_ode_step(w, x, dx, theta, metric_fn, spherical=False):
    """Evolves vector w along path segment dx."""

    # Reference velocity for connection (direction of path)
    v_ref = jax.lax.select(jnp.linalg.norm(dx) > 1e-9, dx, jnp.ones_like(dx) * 1e-5)

    # Ambient Berwald Connection
    # N^i_k = Gamma^i_{jk} dx^j
    N = compute_berwald_connection(theta, x, dx, metric_fn)
    dw = -jnp.dot(N, w)

    # Spherical Correction (Only for R3 -> S2 embedding)
    if spherical:
        dw_sphere = -jnp.dot(w, dx) * x
        dw = dw + dw_sphere

    return dw


# --- 3. PUBLIC INTERFACE ---


# FIX: metric_fn is arg 3, spherical is arg 4. Both must be static.
@partial(jax.jit, static_argnums=(3, 4))
def parallel_transport(theta, path, v_init, metric_fn, spherical=False):
    """
    Finslerian Parallel Transport via Berwald Connection.
    Args:
        theta: Metric params
        path: Sequence of points
        v_init: Start vector
        metric_fn: Function f(theta, x) -> (g, beta)
        spherical: If True, enforces S2 projection (use only for R3 embedding)
    """

    def scan_fn(v_curr, i):
        x_curr = path[i]
        x_next = path[i + 1]
        dx = x_next - x_curr

        dv = transport_ode_step(v_curr, x_curr, dx, theta, metric_fn, spherical=spherical)
        v_next = v_curr + dv

        return v_next, v_next

    num_steps = path.shape[0] - 1
    indices = jnp.arange(num_steps)
    v_final, _ = jax.lax.scan(scan_fn, v_init, indices)

    return v_final
