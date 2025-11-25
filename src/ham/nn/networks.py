import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple


class MetricNet(eqx.Module):
    """
    A Neural Field that learns the geometry of the manifold.
    f(x) -> (raw_L, raw_W)
    """

    layers: list
    split_idx: int

    def __init__(self, key, input_dim, output_dim, hidden_dim=64):
        k1, k2, k3 = jax.random.split(key, 3)
        self.split_idx = output_dim

        # Outputs: L (dim) + W (dim) = 2 * output_dim
        total_out = output_dim * 2

        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=k1),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=k2),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Linear(hidden_dim, total_out, key=k3),
        ]

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        for layer in self.layers:
            x = layer(x)

        raw_L = x[: self.split_idx]
        raw_W = x[self.split_idx :]
        return raw_L, raw_W


class ContextNet(eqx.Module):
    """
    A Conditioned Neural Field.
    f(x, context_id) -> (raw_L, raw_W)
    """

    embedding: eqx.nn.Embedding
    backbone: MetricNet

    def __init__(self, key, vocab_size, context_dim, manifold_dim=3, hidden_dim=64):
        k1, k2 = jax.random.split(key, 2)

        self.embedding = eqx.nn.Embedding(vocab_size, context_dim, key=k1)

        # Backbone Input: ManifoldDim + ContextDim
        # Backbone Output: ManifoldDim (for L and W)
        self.backbone = MetricNet(
            k2, input_dim=manifold_dim + context_dim, output_dim=manifold_dim, hidden_dim=hidden_dim
        )

    def __call__(self, x: jnp.ndarray, context_id: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ctx_vec = self.embedding(context_id)
        combined = jnp.concatenate([x, ctx_vec])
        return self.backbone(combined)


class WindNet(eqx.Module):
    """
    A specialized Neural Field that ONLY learns the Wind (W).
    The Riemannian metric (L) is assumed to be fixed/static.
    """

    layers: list

    def __init__(self, key, input_dim=3, hidden_dim=64):
        k1, k2, k3 = jax.random.split(key, 3)

        # Output dim is just input_dim (Vector field)
        self.layers = [
            eqx.nn.Linear(input_dim, hidden_dim, key=k1),
            eqx.nn.Lambda(jax.nn.tanh),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=k2),
            eqx.nn.Lambda(jax.nn.tanh),
            # Output is just raw_W (3 dims), not 6 dims
            eqx.nn.Linear(hidden_dim, input_dim, key=k3),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x  # Returns raw_W
