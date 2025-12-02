import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
import jax
import jax.numpy as jnp
from jax import grad
import numpy as np

# Import the environment
from ham.sim import coriolis_env
from ham.sim.coriolis_env import EnvParams, EnvState


class TestCoriolisEnv(unittest.TestCase):

    def setUp(self):
        # Force CPU for precision in logic tests
        jax.config.update("jax_platform_name", "cpu")
        self.key = jax.random.PRNGKey(42)
        self.params = EnvParams(dt=0.1, friction=0.0)  # No friction for clean F=ma checks

    def test_gravity_switch(self):
        """
        Verifies that Room 0 pulls South and Room 1 pulls West.
        """
        print("\n--- Test 1: Gravity Switch (Room ID Check) ---")

        # 1. Setup a stationary cargo in the center
        center = jnp.array([5.0, 5.0])
        vel = jnp.zeros(2)

        # 2. Test Room 0 (South Gravity)
        state_0 = EnvState(pos=center, vel=vel, room_id=0, target_pos=center)
        next_0 = coriolis_env.step(state_0, jnp.zeros(2), self.params)

        # Delta Velocity should be [0, -g*dt]
        delta_v0 = next_0.vel - state_0.vel
        print(f"Room 0 Delta V: {delta_v0}")

        self.assertTrue(delta_v0[1] < -0.01, "Room 0 did not pull South!")
        self.assertTrue(abs(delta_v0[0]) < 1e-5, "Room 0 pulled Sideways?")

        # 3. Test Room 1 (West Gravity)
        state_1 = EnvState(pos=center, vel=vel, room_id=1, target_pos=center)
        next_1 = coriolis_env.step(state_1, jnp.zeros(2), self.params)

        delta_v1 = next_1.vel - state_1.vel
        print(f"Room 1 Delta V: {delta_v1}")

        self.assertTrue(delta_v1[0] < -0.01, "Room 1 did not pull West!")
        self.assertTrue(abs(delta_v1[1]) < 1e-5, "Room 1 pulled Vertically?")

        print("SUCCESS: Physics engine respects Room ID.")

    def test_wall_collision(self):
        """
        Verifies that objects don't fly out of the universe.
        """
        print("\n--- Test 2: Wall Constraints ---")

        # Place object right at the left wall (x=0.5) moving LEFT fast
        edge_pos = jnp.array([0.5, 5.0])
        fast_left = jnp.array([-10.0, 0.0])

        state = EnvState(pos=edge_pos, vel=fast_left, room_id=0, target_pos=edge_pos)

        # Step forward
        next_state = coriolis_env.step(state, jnp.zeros(2), self.params)

        print(f"Pos Before: {edge_pos[0]}, Vel Before: {fast_left[0]}")
        print(f"Pos After:  {next_state.pos[0]}, Vel After:  {next_state.vel[0]}")

        # 1. Check Constraint (Should not be negative)
        self.assertTrue(next_state.pos[0] >= 0.5, "Object leaked through wall!")

        # 2. Check Inelastic Collision (Velocity should be killed)
        self.assertEqual(next_state.vel[0], 0.0, "Velocity did not zero out on impact!")

    def test_renderer_differentiability(self):
        """
        CRITICAL: Can we backpropagate from Pixels to Position?
        If this fails, we cannot train a VAE or World Model end-to-end.
        """
        print("\n--- Test 3: Renderer Differentiability ---")

        state = coriolis_env.reset(self.key, room_id=0, params=self.params)

        # Define a loss function on the image
        # e.g. "Maximize brightness of the top-left corner"
        def loss_fn(p):
            # Create state with perturbed position 'p'
            s = EnvState(pos=p, vel=state.vel, room_id=state.room_id, target_pos=state.target_pos)
            img = coriolis_env.render(s, self.params)
            return jnp.sum(img)  # Simple sum loss

        # Compute gradient w.r.t position
        grad_fn = jax.jit(jax.grad(loss_fn))
        grads = grad_fn(state.pos)

        print(f"Gradient of Image w.r.t Position: {grads}")

        # Gradient should be non-zero (changing position changes pixels)
        # and non-NaN.
        self.assertFalse(jnp.isnan(grads).any(), "Gradient is NaN!")
        # Note: Gradient might be small depending on where the object is relative to grid,
        # but with the soft Gaussian renderer, it should generally be non-zero.
        self.assertTrue(
            jnp.linalg.norm(grads) > 0.0, "Gradient is Zero! Rendering is not differentiable."
        )

        print("SUCCESS: Gradients flow through the pixel generation.")


if __name__ == "__main__":
    unittest.main()
