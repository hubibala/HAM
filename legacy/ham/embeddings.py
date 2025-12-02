import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from dataclasses import dataclass
import equinox as eqx


@register_dataclass
@dataclass(frozen=True)
class TokenMap:
    """
    Static mapping of Discrete Tokens <-> Manifold Coordinates.
    Uses a Fibonacci Lattice to distribute tokens evenly on the sphere.
    """

    embedding_matrix: jnp.ndarray  # (Vocab, 3)
    vocab_size: int
    radius: float

    @classmethod
    def create(cls, vocab_size: int, radius: float = 1.0):
        """Factory method to initialize the lattice."""
        indices = jnp.arange(0, vocab_size, dtype=jnp.float32)

        # Fibonacci Lattice formulas
        phi = jnp.arccos(1 - 2 * indices / vocab_size)
        golden_ratio = (1 + 5**0.5) / 2
        theta = 2 * jnp.pi * indices / golden_ratio

        x = jnp.cos(theta) * jnp.sin(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(phi)

        embedding_matrix = jnp.stack([x, y, z], axis=1) * radius

        return cls(embedding_matrix, vocab_size, radius)

    def get_coords(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds a batch of token IDs into manifold coordinates.
        Args:
            token_ids: Integer array of shape (Batch,) or (Batch, Seq)
        Returns:
            coords: Array of shape (..., 3)
        """
        return self.embedding_matrix[token_ids]

    def get_nearest_token(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Decodes manifold coordinates back to the nearest token ID.
        Args:
            coords: Array of shape (..., 3)
        Returns:
            token_ids: Integer array
        """
        # Normalize query coords for cosine similarity
        coords_norm = coords / jnp.linalg.norm(coords, axis=-1, keepdims=True)

        # Normalize embedding matrix (it should be already, but safety first)
        emb_norm = self.embedding_matrix / self.radius

        # Cosine Similarity: x . y
        scores = jnp.dot(coords_norm, emb_norm.T)

        return jnp.argmax(scores, axis=-1)


class LearnableTokenMap(eqx.Module):
    """
    A Trainable Embedding Layer on the Manifold.
    Maps Token ID -> Coordinate.

    Unlike the static TokenMap, this one learns optimal positions.
    """

    embedding: jnp.ndarray  # (Vocab, 3) - Unnormalized parameters

    def __init__(self, key, vocab_size, radius=1.0):
        # Initialize random positions on sphere
        # We use standard normal so the direction is uniform
        self.embedding = jax.random.normal(key, (vocab_size, 3)) * radius

    def __call__(self, token_ids):
        """
        Returns Projected (On-Manifold) Coordinates.
        We normalize in the forward pass to ensure constraints.
        """
        raw_vecs = self.embedding[token_ids]
        norms = jnp.linalg.norm(raw_vecs, axis=-1, keepdims=True)
        return raw_vecs / (norms + 1e-9)

    @property
    def all_coords(self):
        """Returns all normalized coordinates for visualization."""
        norms = jnp.linalg.norm(self.embedding, axis=-1, keepdims=True)
        return self.embedding / (norms + 1e-9)
