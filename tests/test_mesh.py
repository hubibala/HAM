import jax
import jax.numpy as jnp
import unittest
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from ham.geometry.mesh import TriangularMesh

class TestMeshManifold(unittest.TestCase):
    
    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def test_standard_3d_tetrahedron(self):
        """Standard 3D test case."""
        verts = jnp.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
        faces = jnp.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
        mesh = TriangularMesh(verts, faces)
        
        # Project off-surface point
        p = jnp.array([0.2, 0.2, -0.5])
        proj = mesh.project(p)
        expected = jnp.array([0.2, 0.2, 0.0])
        np.testing.assert_allclose(proj, expected, atol=1e-5)
        
        # Tangent vector projection
        x = jnp.array([0.2, 0.2, 0.0]) # On XY face
        v = jnp.array([1.0, 1.0, 1.0]) # Has Z component
        v_tan = mesh.to_tangent(x, v)
        # Should remove Z component
        np.testing.assert_allclose(v_tan, jnp.array([1.0, 1.0, 0.0]), atol=1e-5)

    def test_high_dim_embedding(self):
        """
        Verify mesh logic works for a 2D triangle embedded in 4D.
        Triangle vertices:
        A = [0, 0, 0, 0]
        B = [1, 0, 0, 0]
        C = [0, 1, 0, 0]
        This triangle lies in the X-Y plane of R^4.
        """
        verts = jnp.array([
            [0., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.]
        ])
        faces = jnp.array([[0, 1, 2]])
        mesh = TriangularMesh(verts, faces)
        
        self.assertEqual(mesh.ambient_dim, 4)
        
        # 1. Project point from [0.5, 0.5, 0, 10] -> [0.5, 0.5, 0, 0]
        p_off = jnp.array([0.2, 0.2, 0.0, 10.0])
        proj = mesh.project(p_off)
        expected = jnp.array([0.2, 0.2, 0.0, 0.0])
        np.testing.assert_allclose(proj, expected, atol=1e-5)
        
        # 2. Tangent Projection
        # Vector v = [1, 1, 1, 1] at x on surface.
        # Should keep X, Y components, kill Z, W components.
        v = jnp.array([1.0, 1.0, 1.0, 1.0])
        v_tan = mesh.to_tangent(expected, v)
        np.testing.assert_allclose(v_tan, jnp.array([1.0, 1.0, 0.0, 0.0]), atol=1e-5)
        
        # 3. Sampling
        samples = mesh.random_sample(self.key, (10,))
        self.assertEqual(samples.shape, (10, 4))
        # Ensure samples are in X-Y plane (z=0, w=0)
        np.testing.assert_allclose(samples[:, 2:], 0.0, atol=1e-5)

if __name__ == '__main__':
    unittest.main()