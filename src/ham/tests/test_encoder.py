import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import numpy as np
import os

# Force CPU for stable testing (avoids Metal experimental warnings)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Import the module
from ham.models.encoder import init_encoder_params, apply_encoder, contrastive_loss


class TestSphericalEncoder(unittest.TestCase):

    def setUp(self):
        self.key = random.PRNGKey(42)
        self.img_shape = (64, 64, 1)  # H, W, C

        # Initialize parameters
        self.params = init_encoder_params(self.key)

    def test_spherical_constraint(self):
        """
        CRITICAL: The output embedding MUST lie on the unit sphere.
        If ||z|| != 1, the Finsler metric (defined on S2) will be invalid.
        """
        print("\n--- Test 1: Spherical Constraint ---")

        # Create random dummy image
        dummy_img = random.normal(self.key, self.img_shape)

        # Forward pass
        embedding = apply_encoder(self.params, dummy_img)

        print(f"Embedding Vector: {embedding}")
        norm = jnp.linalg.norm(embedding)
        print(f"Norm: {norm}")

        # Check norm is 1.0 (within float precision)
        self.assertTrue(jnp.allclose(norm, 1.0, atol=1e-5), "Encoder failed to project to Sphere!")

    def test_batch_processing(self):
        """
        Verifies that the encoder handles batches correctly.
        HAM requires (Batch, 3) output.
        """
        print("\n--- Test 2: Batch Processing ---")

        batch_size = 10
        batch_shape = (batch_size,) + self.img_shape
        dummy_batch = random.normal(self.key, batch_shape)

        embeddings = apply_encoder(self.params, dummy_batch)

        print(f"Output Shape: {embeddings.shape}")

        self.assertEqual(embeddings.shape, (batch_size, 3))

        # Check norms of ALL items in batch
        norms = jnp.linalg.norm(embeddings, axis=1)
        self.assertTrue(
            jnp.allclose(norms, 1.0, atol=1e-5), "Batch projection failed constraint check."
        )

    def test_contrastive_loss_flow(self):
        """
        Verifies that we can compute gradients of the Contrastive Loss.
        This proves the 'Perception Layer' is trainable.
        """
        print("\n--- Test 3: Learning Signal (Gradient Flow) ---")

        # Create a fake trajectory: (Batch=2, Time=5, H=64, W=64, C=1)
        # This simulates 2 episodes of 5 frames each.
        B, T = 2, 5
        traj_shape = (B, T) + self.img_shape
        dummy_traj = random.normal(self.key, traj_shape)

        # Define gradient function
        loss_fn = lambda p: contrastive_loss(p, dummy_traj)
        grad_fn = jit(grad(loss_fn))

        # Compute gradients
        grads = grad_fn(self.params)

        # Check gradient structure matches params
        self.assertEqual(grads.keys(), self.params.keys())

        # Check a specific layer (e.g., Convolution 1) to ensure signal reaches input
        c1_grad_w, c1_grad_b = grads["c1"]
        grad_norm = jnp.linalg.norm(c1_grad_w)

        print(f"Conv1 Weight Gradient Norm: {grad_norm}")

        self.assertFalse(jnp.isnan(grad_norm), "Gradients represent NaN!")
        self.assertTrue(grad_norm > 0.0, "Gradients are Zero! Loss is disconnected.")

        print("SUCCESS: Perception layer is differentiable.")


if __name__ == "__main__":
    unittest.main()
