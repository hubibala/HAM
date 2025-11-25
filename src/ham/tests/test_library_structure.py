import os
import sys
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Force JAX to use CPU only
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
import ham

def test_library_imports():
    print("--- Testing HAM Library Structure ---")
    
    # 1. Test Mesh Gen
    print("Testing Mesh Generation...")
    verts, faces = ham.generate_icosphere(subdivisions=2)
    print(f"Mesh created: {verts.shape} vertices, {faces.shape} faces")
    assert verts.shape == (162, 3)
    
    # 2. Test Embeddings
    print("Testing TokenMap...")
    vocab_size = 100
    tmap = ham.TokenMap.create(vocab_size)
    
    # Check lookup
    ids = jnp.array([0, 10, 99])
    coords = tmap.get_coords(ids)
    print(f"Looked up 3 tokens: {coords.shape}")
    assert coords.shape == (3, 3)
    
    # Check reverse lookup
    rec_ids = tmap.get_nearest_token(coords)
    print(f"Recovered IDs: {rec_ids}")
    assert jnp.array_equal(ids, rec_ids)
    
    print("SUCCESS: Library is integrated.")

if __name__ == "__main__":
    test_library_imports()