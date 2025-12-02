import jax
import jax.numpy as jnp
import unittest
import numpy as np
from jax import config

# Ensure high precision
config.update("jax_enable_x64", True)

from ham.geometry.manifold import Manifold
from ham.geometry.zoo import Euclidean, Randers
from ham.solvers.avbd import AVBDSolver

# --- Mock Manifolds (Re-declared for standalone testing context) ---
class FlatPlane(Manifold):
    @property
    def ambient_dim(self): return 2
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def random_sample(self, key, shape): 
        return jax.random.normal(key, shape + (2,))

class Sphere(Manifold):
    @property
    def ambient_dim(self): return 3
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x / jnp.linalg.norm(x)
    def to_tangent(self, x, v):
        x_unit = x / jnp.linalg.norm(x)
        return v - jnp.dot(v, x_unit) * x_unit
    def random_sample(self, key, shape):
        x = jax.random.normal(key, shape + (3,))
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

class TestAVBDSolver(unittest.TestCase):
    
    def setUp(self):
        self.plane = FlatPlane()
        self.sphere = Sphere()
        self.key = jax.random.PRNGKey(42)
        # Default solver configuration
        self.solver = AVBDSolver(step_size=0.01, max_iter=300)

    def test_euclidean_straight_line(self):
        """In flat space, the geodesic must be a straight line."""
        metric = Euclidean(self.plane)
        start, end = jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])
        
        traj = self.solver.solve(metric, start, end, n_steps=10)
        
        # Check linearity (Midpoint should be (0.5, 0.5))
        mid_point = traj.xs[5]
        np.testing.assert_allclose(mid_point, jnp.array([0.5, 0.5]), atol=1e-2)

    def test_zermelo_ferry_crossing(self):
        """
        Verifies the Zermelo Navigation problem (The Ferry Problem).
        
        Scenario: 
            - Constant Wind W = [0.5, 0] (Flowing East).
            - We compare the energy cost of three distinct trajectories:
              1. Downwind (West -> East)
              2. Crosswind (South -> North)
              3. Upwind (East -> West)
              
        Expected Physics:
            Energy(Downwind) < Energy(Crosswind) < Energy(Upwind)
        """
        # 1. Setup Randers Metric (Wind flows +X)
        h_net = lambda x: jnp.eye(2)
        w_net = lambda x: jnp.array([0.5, 0.0]) 
        metric = Randers(self.plane, h_net, w_net)
        
        # 2. Define equidistant paths (Euclidean length = 2.0)
        # Path A: Downwind (0,0) -> (2,0)
        traj_down = self.solver.solve(metric, jnp.array([0.0,0.0]), jnp.array([2.0,0.0]))
        
        # Path B: Crosswind (0,0) -> (0,2)
        traj_cross = self.solver.solve(metric, jnp.array([0.0,0.0]), jnp.array([0.0,2.0]))
        
        # Path C: Upwind (2,0) -> (0,0)
        traj_up = self.solver.solve(metric, jnp.array([2.0,0.0]), jnp.array([0.0,0.0]))
        
        # 3. Assert Geometric Constraints
        # Crosswind path should stay roughly vertical (x~0) despite wind
        cross_deviation = jnp.abs(traj_cross.xs[:, 0]).max()
        self.assertLess(cross_deviation, 0.1, "Path deviated significantly from straight line.")

        # 4. Assert Energy Asymmetry (The core Finsler property)
        # Downwind cost < Crosswind cost < Upwind cost
        self.assertLess(traj_down.energy, traj_cross.energy, "Downwind should be cheaper than Crosswind")
        self.assertLess(traj_cross.energy, traj_up.energy, "Crosswind should be cheaper than Upwind")

    def test_sphere_constraints(self):
        """Solver must not tunnel through the sphere."""
        metric = Euclidean(self.sphere)
        start = jnp.array([0.0, 0.0, 1.0]) # North Pole
        end = jnp.array([1.0, 0.0, 0.0])   # Equator
        
        traj = self.solver.solve(metric, start, end, n_steps=10)
        
        # Check norms strictly equal 1.0
        norms = jnp.linalg.norm(traj.xs, axis=1)
        np.testing.assert_allclose(norms, jnp.ones_like(norms), atol=1e-4)
    
    def test_short_path_edge_case(self):
        """
        What happens if we request only 2 steps?
        Should just be Start -> End.
        """
        metric = Euclidean(self.plane)
        traj = self.solver.solve(metric, jnp.zeros(2), jnp.ones(2), n_steps=1)
        
        # Trajectory has n_steps + 1 points = 2 points
        self.assertEqual(traj.xs.shape[0], 2)
        np.testing.assert_allclose(traj.xs[0], jnp.zeros(2))
        np.testing.assert_allclose(traj.xs[1], jnp.ones(2))

if __name__ == '__main__':
    unittest.main()