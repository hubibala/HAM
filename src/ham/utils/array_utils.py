"""Utility functions for safe array conversions"""
import numpy as np
import jax
import jax.numpy as jnp


def to_jax_array(arr, dtype=None):
    """
    Safely convert a numpy array to a JAX array.
    
    This function ensures compatibility with different JAX backends
    by first converting to a standard numpy array.
    
    Args:
        arr: Input array (numpy, list, or JAX array)
        dtype: Optional dtype for the output array
        
    Returns:
        JAX array
    """
    # First ensure it's a numpy array
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    
    # Convert to JAX using device_put which is more reliable
    if dtype is not None:
        arr = arr.astype(dtype)
    
    return jax.device_put(arr)


def ensure_cpu():
    """
    Ensure JAX is using CPU backend.
    Call this at the start of scripts/notebooks if you need CPU-only execution.
    """
    import os
    os.environ['JAX_PLATFORMS'] = 'cpu'
    # Force JAX to reinitialize with the new platform
    jax.config.update('jax_platform_name', 'cpu')
