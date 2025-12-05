import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List

class MLP(eqx.Module):
    """Standard Multi-Layer Perceptron using Equinox."""
    layers: List[eqx.nn.Linear]
    activation: Callable

    def __init__(self, in_size, out_size, width_size, depth, key, activation=jax.nn.softplus):
        # Softplus is often smoother for geometry than ReLU
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        self.layers.append(eqx.nn.Linear(in_size, width_size, key=keys[0]))
        for i in range(depth - 1):
            self.layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))
        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

class VectorField(eqx.Module):
    """
    Learns a vector field W(x): R^N -> R^N.
    Used for the wind field in Randers metrics.
    """
    net: MLP

    def __init__(self, ambient_dim, hidden_dim, depth, key):
        self.net = MLP(ambient_dim, ambient_dim, hidden_dim, depth, key)

    def __call__(self, x):
        return self.net(x)

class PSDMatrixField(eqx.Module):
    """
    Learns a Positive Semi-Definite matrix field G(x): R^N -> S+(N).
    
    The network outputs a raw matrix A(x).
    The metric is constructed as G(x) = A(x) @ A(x).T + epsilon * I.
    This guarantees valid Riemannian metrics by construction.
    """
    net: MLP
    ambient_dim: int
    epsilon: float

    def __init__(self, ambient_dim, hidden_dim, depth, key, epsilon=1e-4):
        self.ambient_dim = ambient_dim
        self.epsilon = epsilon
        # Output dimension is N*N (flattened matrix)
        self.net = MLP(ambient_dim, ambient_dim * ambient_dim, hidden_dim, depth, key)

    def __call__(self, x):
        # 1. Get raw output (N*N)
        raw = self.net(x)
        
        # 2. Reshape to (N, N)
        A = raw.reshape(self.ambient_dim, self.ambient_dim)
        
        # 3. Construct PSD matrix: G = A @ A.T + eps * I
        G = jnp.dot(A, A.T) + self.epsilon * jnp.eye(self.ambient_dim)
        
        return G