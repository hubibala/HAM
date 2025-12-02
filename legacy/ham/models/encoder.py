import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import optax

# --- 1. ARCHITECTURE (Raw JAX CNN) ---


def init_encoder_params(key, output_dim=3):
    """
    Initializes a simple 3-layer ConvNet + MLP Head.
    Input: (64, 64, 1) -> Output: (output_dim,) normalized to S^n
    """
    k1, k2, k3, k4, k5 = random.split(key, 5)

    # Helper for initializing layers
    def conv_layer(k, c_in, c_out, kernel=3):
        w = random.normal(k, (kernel, kernel, c_in, c_out)) * jnp.sqrt(2 / (kernel * kernel * c_in))
        b = jnp.zeros(c_out)
        return w, b

    def dense_layer(k, d_in, d_out):
        w = random.normal(k, (d_in, d_out)) * jnp.sqrt(2 / d_in)
        b = jnp.zeros(d_out)
        return w, b

    # Conv 1: 1 -> 8 (Stride 2)
    c1_w, c1_b = conv_layer(k1, 1, 8, kernel=4)
    # Conv 2: 8 -> 16 (Stride 2)
    c2_w, c2_b = conv_layer(k2, 8, 16, kernel=4)
    # Conv 3: 16 -> 32 (Stride 2)
    c3_w, c3_b = conv_layer(k3, 16, 32, kernel=4)

    # Dense Head
    d1_w, d1_b = dense_layer(k4, 8 * 8 * 32, 128)
    d2_w, d2_b = dense_layer(k5, 128, output_dim)  # Dynamic Output Size

    return {
        "c1": (c1_w, c1_b),
        "c2": (c2_w, c2_b),
        "c3": (c3_w, c3_b),
        "d1": (d1_w, d1_b),
        "d2": (d2_w, d2_b),
    }


def apply_encoder(params, x):
    """
    Forward pass.
    x: (Batch, 64, 64, 1) or (64, 64, 1)
    Returns: (Batch, dim) or (dim,) on Unit Sphere
    """
    is_batched = x.ndim == 4
    if not is_batched:
        x = x[None, ...]

    dn = ("NHWC", "HWIO", "NHWC")

    # Conv 1
    w, b = params["c1"]
    x = jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=dn)
    x = jax.nn.relu(x + b)

    # Conv 2
    w, b = params["c2"]
    x = jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=dn)
    x = jax.nn.relu(x + b)

    # Conv 3
    w, b = params["c3"]
    x = jax.lax.conv_general_dilated(x, w, (2, 2), "SAME", dimension_numbers=dn)
    x = jax.nn.relu(x + b)

    # Flatten
    x = x.reshape((x.shape[0], -1))

    # MLP Head
    w, b = params["d1"]
    x = jax.nn.relu(jnp.dot(x, w) + b)

    w, b = params["d2"]
    embedding = jnp.dot(x, w) + b

    # Spherical Projection
    norm = jnp.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / (norm + 1e-6)

    if not is_batched:
        return embedding[0]
    return embedding


# --- 2. CONTRASTIVE LEARNING (Temporal) ---


def contrastive_loss(params, batch_trajectory):
    """
    Time-Contrastive Loss.
    batch_trajectory: (Batch, T, 64, 64, 1)
    """
    # 1. Encode all frames
    B, T, H, W, C = batch_trajectory.shape
    flat_imgs = batch_trajectory.reshape(B * T, H, W, C)

    embeddings = apply_encoder(params, flat_imgs)  # (B*T, dim)

    # FIX: Infer dimension dynamically instead of hardcoding 3
    dim = embeddings.shape[-1]

    # Reshape back
    z = embeddings.reshape(B, T, dim)

    # 2. Define Loss
    # Pairs: (t, t+1)
    z_current = z[:, :-1, :]  # (B, T-1, dim)
    z_next = z[:, 1:, :]  # (B, T-1, dim)

    # Alignment Loss: - ||z_t - z_{t+1}||^2
    loss_align = jnp.mean(jnp.sum((z_current - z_next) ** 2, axis=2))

    # Uniformity Loss (Regularization)
    # Limit to first 128 to save compute
    z_sub = embeddings[:128]
    sim_matrix = jnp.dot(z_sub, z_sub.T)
    loss_uniform = jnp.mean(sim_matrix**2)

    return loss_align + 0.5 * loss_uniform
