import jax
import jax.numpy as jnp

def safe_norm(x, axis=-1, keepdims=False, eps=1e-12):
    """Computes norm(x) safely, avoiding NaN gradients at x=0."""
    is_zero = jnp.all(x == 0, axis=axis, keepdims=True)
    # Replace zeros with ones temporarily to compute safe gradient
    # The actual value at 0 will be masked out, but the branch protects the gradient
    x_safe = jnp.where(is_zero, jnp.ones_like(x), x)
    return jnp.where(
        is_zero.squeeze() if not keepdims else is_zero,
        0.0,
        jnp.linalg.norm(x_safe, axis=axis, keepdims=keepdims)
    )