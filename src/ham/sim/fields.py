import jax
import jax.numpy as jnp

def get_stream_function_flow(psi_fn):
    """
    Converts a scalar stream function psi(x, y, z) into a vector field.
    v = grad(psi) x position
    """
    def flow(x):
        grad_psi = jax.grad(psi_fn)(x)
        # v = n x grad(psi)
        return jnp.cross(x, grad_psi)
    return flow

def tilted_rotation(alpha_deg=45.0):
    alpha = jnp.radians(alpha_deg)
    axis = jnp.array([jnp.sin(alpha), 0.0, jnp.cos(alpha)])
    
    def psi(x):
        return jnp.dot(axis, x)
        
    return get_stream_function_flow(psi)

def rossby_haurwitz(R=4, omega=1.0, K=1.0):
    """
    The classic wavy planetary flow.
    """
    def psi(x):
        # sin(lat) = z
        sin_lat = x[2]
        
        # cos(lat) = sqrt(x^2 + y^2)
        # Epsilon for gradient safety
        cos_lat = jnp.sqrt(x[0]**2 + x[1]**2 + 1e-12)
        
        # cos(R * lon) via complex arithmetic
        xy_complex = x[0] + 1j * x[1]
        # Safe normalization
        xy_unit = xy_complex / (jnp.abs(xy_complex) + 1e-12)
        cos_R_lon = jnp.real(xy_unit ** R)
        
        term1 = -omega * sin_lat
        term2 = K * (cos_lat ** R) * sin_lat * cos_R_lon
        
        return term1 + term2

    return get_stream_function_flow(psi)

def harmonic_vortices(l=5, m=3):
    """
    High-frequency cellular flow.
    """
    def psi(x):
        z = x[2]
        cos_lat_m = (1.0 - z**2 + 1e-12)**(m / 2.0)
        
        xy_complex = x[0] + 1j * x[1]
        xy_unit = xy_complex / (jnp.abs(xy_complex) + 1e-12)
        cos_m_lon = jnp.real(xy_unit ** m)
        
        z_poly = jnp.sin(3.0 * jnp.pi * z) 
        
        return cos_lat_m * z_poly * cos_m_lon

    return get_stream_function_flow(psi)