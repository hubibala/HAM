import jax
import jax.numpy as jnp
import unittest
import numpy as np
import equinox as eqx
from ham.nn.networks import VectorField, PSDMatrixField

class TestNeuralFields(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)
        self.dim = 3

    def test_vector_field_shape(self):
        """Vector field should output (D,) for input (D,)."""
        vf = VectorField(self.dim, 16, 2, self.key)
        x = jnp.ones(self.dim)
        v = vf(x)
        self.assertEqual(v.shape, (self.dim,))

    def test_psd_matrix_properties(self):
        """Matrix field must be Symmetric and Positive Definite."""
        psd_net = PSDMatrixField(self.dim, 16, 2, self.key)
        x = jnp.array([0.5, -0.2, 1.0])
        
        G = psd_net(x)
        
        # 1. Shape Check
        self.assertEqual(G.shape, (self.dim, self.dim))
        
        # 2. Symmetry Check
        np.testing.assert_allclose(G, G.T, atol=1e-5)
        
        # 3. Positive Definiteness Check (Eigenvalues > 0)
        eigs = jnp.linalg.eigvalsh(G)
        self.assertTrue(jnp.all(eigs > 0), "Matrix field output is not positive definite!")

if __name__ == '__main__':
    unittest.main()