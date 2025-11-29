import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn


class RaceCarEncoder(nn.Module):
    """
    Maps (96, 96, 9) image stacks -> S^2 Manifold.
    Uses Spatial Attention to focus on track geometry (Ice/Road/Wall).
    """

    latent_dim: int = 3  # The Sphere S^2

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Input x: (Batch, 96, 96, 9)

        # 1. Perception Backbone (CNN)
        # Stride 2 reduces dimension quickly: 96 -> 48 -> 24 -> 12
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.relu(x)  # Shape: (Batch, 12, 12, 128)

        # 2. Spatial Attention (The "Fovea")
        # Generates a heatmap (12x12) of importance
        attn_logits = nn.Conv(features=1, kernel_size=(1, 1))(x)

        # Softmax over spatial dimensions
        B, H, W, C = attn_logits.shape
        flat_attn = attn_logits.reshape(B, -1)
        weights = jax.nn.softmax(flat_attn, axis=-1).reshape(B, H, W, C)

        # Apply attention mask
        x = x * weights

        # 3. Geometric Projection
        x = x.reshape((B, -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        # Output unnormalized vector
        embedding = nn.Dense(self.latent_dim)(x)

        # 4. Manifold Constraint (S^2)
        # Project onto unit sphere
        norm = jnp.linalg.norm(embedding, axis=-1, keepdims=True)
        return embedding / (norm + 1e-6)
