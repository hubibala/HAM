import jax
import jax.numpy as jnp
from jax import grad, jit


def holonomy_error_loss(
    theta, p_source, v_source, p_target, v_target_true, metric_fn, solver_fn, transport_fn
):
    """
    Computes the Holonomic Discrepancy Loss between a guessed skill transport
    and the true required skill in a new context.
    """

    # 1. Solve for the Geodesic (The "Symmetry" Path)
    # The solver finds the optimal path connecting the two contexts based on
    # the CURRENT understanding of the geometry (theta).
    path_states = solver_fn(theta, p_source, p_target)

    # 2. Parallel Transport the Skill (The "Zero-Shot Guess")
    # We transport the source skill vector along the discovered geodesic.
    # FIX: We must pass 'metric_fn' here so transport knows the geometry!
    v_guess = transport_fn(theta, path_states, v_source, metric_fn)

    # 3. Compute the Discrepancy (The Error Signal)
    # The error is the difference between the "Guess" and "Reality".
    diff = v_guess - v_target_true

    # 4. Measure Error in the Target's Local Metric
    # We must use the Riemannian metric g at the target to properly measure length.
    g_target, _ = metric_fn(theta, p_target)

    # Loss = 0.5 * <diff, diff>_g
    # Uses einsum for efficient quadratic form calculation: diff^T * G * diff
    loss = 0.5 * jnp.einsum("i,ij,j->", diff, g_target, diff)

    return loss
