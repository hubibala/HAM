import unittest
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from ham.solvers.avbd import AVBDSolver
from ham.geometry.surfaces import Sphere, Torus, Paraboloid
from ham.geometry.zoo import Euclidean, Randers
from ham.utils.math import safe_norm

class TestSolver(unittest.TestCase):
    
    def setUp(self):
        # AVBD Settings: 200 iterations
        self.solver = AVBDSolver(step_size=0.05, iterations=200, tol=1e-6)

    def test_torus_topology(self):
        """
        Geodesic on a Torus must wrap around the tube, not cut through the hole.
        """
        torus = Torus(major_R=2.0, minor_r=1.0)
        metric = Euclidean(torus)
        
        # Start (Outer Equator) -> End (Inner Equator)
        start = jnp.array([3.0, 0.0, 0.0]) # 2 + 1
        end   = jnp.array([1.0, 0.0, 0.0]) # 2 - 1
        
        traj = self.solver.solve(metric, start, end, n_steps=20)
        
        # 1. Check Constraint
        # (sqrt(x^2+y^2) - R)^2 + z^2 approx r^2
        xy = safe_norm(traj.xs[:,:2], axis=1)
        dist = jnp.abs((xy - torus.R)**2 + traj.xs[:,2]**2 - torus.r**2)
        max_err = jnp.max(dist)
        
        print(f"\nTorus Violation: {max_err:.2e}")
        self.assertLess(max_err, 0.1) # Tolerance for discrete path
        
        # 2. Check Topology (Did it go Over?)
        max_z = jnp.max(jnp.abs(traj.xs[:, 2]))
        print(f"Max Height (Z): {max_z:.2f}")
        self.assertGreater(max_z, 0.5, "Path failed to wrap around the Torus tube.")

    def test_sphere_zermelo(self):
        """
        Test zoo.Randers on a Sphere.
        Wind flows +X. Moving East (Downwind) should be cheaper than West (Upwind).
        """
        sphere = Sphere(radius=1.0)
        
        # Define networks for Randers
        # Identity metric, Constant wind [0.5, 0, 0]
        h_net = lambda x: jnp.eye(3)
        w_net = lambda x: jnp.array([0.5, 0.0, 0.0])
        
        metric = Randers(sphere, h_net, w_net)
        
        # Points on Equator
        p_west = jnp.array([-0.5, 0.0, 0.866])
        p_east = jnp.array([ 0.5, 0.0, 0.866])
        
        p_west = sphere.project(p_west)
        p_east = sphere.project(p_east)
        
        # Solve
        traj_down = self.solver.solve(metric, p_west, p_east, n_steps=15)
        traj_up   = self.solver.solve(metric, p_east, p_west, n_steps=15)
        
        print(f"\nSphere Zermelo Energy:\nDownwind: {traj_down.energy:.4f}\nUpwind:   {traj_up.energy:.4f}")
        
        self.assertLess(traj_down.energy, traj_up.energy)

    def test_paraboloid_implicit(self):
        """Test implicit constraint solver on Paraboloid z = x^2 + y^2."""
        para = Paraboloid()
        metric = Euclidean(para)
        
        # Use explicit constraint function for the solver to be sure
        def para_c(x): return x[2] - (x[0]**2 + x[1]**2)
        
        start = jnp.array([-1.0, 0.0, 1.0])
        end   = jnp.array([ 1.0, 0.0, 1.0])
        
        traj = self.solver.solve(metric, start, end, n_steps=20, constraints=[para_c])
        
        mid = traj.xs[10]
        print(f"\nParaboloid Midpoint Z: {mid[2]:.4f}")
        self.assertLess(mid[2], 0.2, "Path did not dip to follow surface.")

    def test_rk4_ode(self):
        """Test RK4 ODE integration for simple circular dynamics."""
        # Simple harmonic oscillator / circular dynamics
        def vector_field(z):
            return jnp.array([-z[1], z[0]])
            
        z0 = jnp.array([1.0, 0.0])
        step_size = 0.01
        max_steps = int(jnp.pi * 2 / step_size)
        
        # RK4
        def scan_fn(z, _):
            k1 = vector_field(z)
            k2 = vector_field(z + step_size/2 * k1)
            k3 = vector_field(z + step_size/2 * k2)
            k4 = vector_field(z + step_size * k3)
            z_next = z + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
            return z_next, z_next
            
        _, traj = jax.lax.scan(scan_fn, z0, jnp.arange(max_steps))
        traj = jnp.concatenate([z0[None], traj], axis=0)
        
        # After 2*pi, we should be back at [1.0, 0.0]
        final_pt = traj[-1]
        error = jnp.linalg.norm(final_pt - z0)
        print(f"\nRK4 Circular Dynamics Error: {error:.4e}")
        self.assertLess(error, 1e-2, "RK4 Integration drifted too much on circle.")

if __name__ == '__main__':
    unittest.main()