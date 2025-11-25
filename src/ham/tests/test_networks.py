import os
import sys
# Ensure we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Force JAX to use CPU only
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from ham.manifolds import Sphere
from ham.geometry import RandersFactory
from ham.nn import MetricNet, ContextNet

def test_metric_net_shapes():
    print("\n--- Testing MetricNet (Shapes & Batching) ---")
    
    key = jax.random.PRNGKey(0)
    manifold_dim = 3 # Ambient dim for S^2
    
    # 1. Initialize
    # Input: 3D point. Output: 3D shape + 3D wind.
    net = MetricNet(key, input_dim=manifold_dim, output_dim=manifold_dim, hidden_dim=32)
    
    # 2. Single Input Test
    x = jnp.array([1.0, 0.0, 0.0])
    raw_L, raw_W = net(x)
    
    print(f"Single Input x: {x.shape}")
    print(f"Output raw_L: {raw_L.shape}, raw_W: {raw_W.shape}")
    
    assert raw_L.shape == (manifold_dim,), "L shape mismatch"
    assert raw_W.shape == (manifold_dim,), "W shape mismatch"
    
    # 3. Batch Input Test (vmap)
    batch_size = 10
    xs = jax.random.normal(key, (batch_size, manifold_dim))
    
    # vmap the network forward pass
    # eqx.filter_vmap is safer for Equinox modules than jax.vmap
    batched_forward = eqx.filter_vmap(net)
    batch_L, batch_W = batched_forward(xs)
    
    print(f"Batch Input xs: {xs.shape}")
    print(f"Batch Output L: {batch_L.shape}, W: {batch_W.shape}")
    
    assert batch_L.shape == (batch_size, manifold_dim)
    assert batch_W.shape == (batch_size, manifold_dim)
    print("SUCCESS: MetricNet handles shapes and batching correctly.")

def test_context_net_shapes():
    print("\n--- Testing ContextNet (Embeddings) ---")
    
    key = jax.random.PRNGKey(1)
    vocab_size = 5
    context_dim = 4
    manifold_dim = 3
    
    # 1. Initialize
    net = ContextNet(key, vocab_size, context_dim, manifold_dim=manifold_dim)
    
    # 2. Batch Test
    batch_size = 8
    xs = jax.random.normal(key, (batch_size, manifold_dim))
    ctx_ids = jax.random.randint(key, (batch_size,), 0, vocab_size)
    
    batched_forward = eqx.filter_vmap(net)
    batch_L, batch_W = batched_forward(xs, ctx_ids)
    
    print(f"Context IDs: {ctx_ids.shape}")
    print(f"Output L: {batch_L.shape}, W: {batch_W.shape}")
    
    assert batch_L.shape == (batch_size, manifold_dim)
    print("SUCCESS: ContextNet correctly fuses position and context.")

def test_physics_integration():
    print("\n--- Testing Integration: Neural Net -> Factory -> Metric ---")
    
    # Can the output of the NN be fed directly into the Physics Engine?
    manifold = Sphere(dim=2) # R3 ambient
    factory = RandersFactory(manifold)
    key = jax.random.PRNGKey(2)
    
    net = MetricNet(key, input_dim=3, output_dim=3)
    
    # Generate Random Points on Sphere
    x = manifold.random_uniform(key, (1,))[0] # Single point
    
    # Forward Pass
    raw_L, raw_W = net(x)
    
    # Create Physics Object
    metric = factory.forward(x, raw_L, raw_W)
    
    print("Metric object created successfully.")
    print(f"Metric a shape: {metric.a.shape}")
    print(f"Metric beta shape: {metric.beta.shape}")
    
    # CRITICAL CHECK: Is the output physically valid?
    # The Network outputs random garbage initially.
    # The Factory MUST sanitize it.
    
    # Check Tangency: <x, beta> should be 0
    radial = jnp.dot(x, metric.beta)
    print(f"Radial component of beta: {radial:.2e}")
    assert jnp.abs(radial) < 1e-5, "Network output failed tangency check!"
    
    # Check Convexity: ||beta||_a < 1
    # beta^T a^-1 beta
    a_inv = jnp.linalg.inv(metric.a)
    norm_sq = metric.beta @ a_inv @ metric.beta
    print(f"Anisotropic Norm of beta: {jnp.sqrt(norm_sq):.4f}")
    assert norm_sq < 1.0, "Network output failed convexity check!"
    
    print("SUCCESS: Neural Network output is successfully sanitized by Factory.")

if __name__ == "__main__":
    test_metric_net_shapes()
    test_context_net_shapes()
    test_physics_integration()