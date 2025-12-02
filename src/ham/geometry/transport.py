import jax
import jax.numpy as jnp
from ham.geometry.metric import FinslerMetric

def berwald_transport(metric: FinslerMetric, 
                      path_x: jnp.ndarray, 
                      path_v: jnp.ndarray, 
                      vec_start: jnp.ndarray) -> jnp.ndarray:
    """
    Parallel transports a vector 'vec_start' along a trajectory (path_x, path_v)
    using the Berwald Connection.
    
    The Berwald connection is induced directly by the Spray G.
    Equation: dX/dt + Jac_v(G)(x, v) * X = 0
    
    Args:
        metric: The Finsler geometry.
        path_x: Shape (T, D) - Position sequence.
        path_v: Shape (T, D) - Velocity sequence (tangent to x).
        vec_start: Shape (D,) - The vector at path_x[0] to transport.
        
    Returns:
        transported_vecs: Shape (T, D) - The vector field along the path.
    """
    
    # We define the ODE derivative function for jax.lax.scan
    def transport_ode(carry_vec, inputs):
        x, v = inputs # Current position and velocity of the base curve
        
        # 1. Compute the Connection Matrix (D, D)
        # For Berwald: Gamma^i_j = d(G^i)/d(v^j)
        # We use Jacobian of the spray function w.r.t velocity (arg 1)
        # Note: G is defined as -0.5 * acceleration. 
        # The ODE is D_t X = 0 => dX/dt + Gamma * X = 0
        jac_spray = jax.jacfwd(metric.spray, argnums=1)(x, v)
        
        # 2. Compute Derivative of the transported vector
        # dX = - Gamma * X
        # The linear connection term for Berwald is Jac_v(G).
        dx = -jnp.dot(jac_spray, carry_vec)
        
        # 3. Update (Euler integration step)
        # We use a normalized dt = 1.0 / Steps because the spray G is already 
        # scaled by the energy function's time assumptions.
        dt = 1.0 / len(path_x)
        new_vec = carry_vec + dx * dt
        
        # 4. Functional Constraint Enforcement
        # Ensure the vector stays tangent to the manifold.
        new_vec = metric.manifold.to_tangent(x, new_vec)
        
        return new_vec, new_vec

    # Run Scan (Efficient Loop)
    _, transported_vecs = jax.lax.scan(
        transport_ode, 
        vec_start, 
        (path_x, path_v)
    )
    
    # The scan outputs the vector *after* each step. 
    # We want the sequence starting from vec_start.
    # We discard the very last projected point to keep shapes consistent (T, D)
    result = jnp.concatenate([vec_start[None, :], transported_vecs[:-1]], axis=0)
    
    return result