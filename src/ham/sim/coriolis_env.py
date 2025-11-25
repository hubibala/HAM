import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Force JAX to use CPU only
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from jax import jit, random, vmap
from jax.tree_util import register_dataclass
from dataclasses import dataclass, field  # <--- Added 'field'
from typing import Tuple

# --- 1. STATE MANAGEMENT ---


@register_dataclass
@dataclass(frozen=True)
class EnvState:
    pos: jnp.ndarray  # (2,) [x, y] in meters
    vel: jnp.ndarray  # (2,) [vx, vy] in m/s
    room_id: int  # 0 = Room A (Training), 1 = Room B (Testing)
    target_pos: jnp.ndarray  # (2,) [x, y]


@register_dataclass
@dataclass(frozen=True)
class EnvParams:
    dt: float = 0.1  # Time step
    mass: float = 1.0  # Mass of cargo
    friction: float = 0.5  # Friction coefficient

    # FIX: Use default_factory for array fields to satisfy dataclass rules
    gravity_A: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, -1.0]))  # Pulls DOWN
    gravity_B: jnp.ndarray = field(default_factory=lambda: jnp.array([0.707, 0.707]))  # Pulls LEFT

    # Rendering
    img_size: int = 64
    world_size: float = 10.0  # Meters


# --- 2. PHYSICS ENGINE ---


def step(state: EnvState, action: jnp.ndarray, params: EnvParams) -> EnvState:
    """
    Evolves the system one step.
    Action: Continuous Force vector (2,) applied by thrusters.
    """
    # 1. Select Global Physics based on Room ID
    gravity = jax.lax.select(state.room_id == 0, params.gravity_A, params.gravity_B)

    # 2. Forces
    # F_net = F_thruster + F_gravity - F_friction
    force = action + gravity - params.friction * state.vel

    # 3. Integration (Semi-Implicit Euler)
    accel = force / params.mass
    new_vel = state.vel + accel * params.dt
    new_pos = state.pos + new_vel * params.dt

    # 4. Boundaries (Bounce)
    lower = 0.5  # Wall thickness buffer
    upper = params.world_size - 0.5

    clipped_pos = jnp.clip(new_pos, lower, upper)
    # If we hit a wall, kill velocity (inelastic collision)
    hit_wall = (new_pos < lower) | (new_pos > upper)
    new_vel = jnp.where(hit_wall, 0.0, new_vel)

    return EnvState(
        pos=clipped_pos, vel=new_vel, room_id=state.room_id, target_pos=state.target_pos
    )


def reset(key: jax.Array, room_id: int = 0, params: EnvParams = None) -> EnvState:
    """Resets the cargo to the center with random noise."""
    # Handle mutable default manually if passed as None
    if params is None:
        params = EnvParams()

    k1, k2 = random.split(key)

    # Start near center
    center = params.world_size / 2.0
    pos = jnp.array([center, center]) + random.normal(k1, (2,)) * 0.5
    vel = jnp.zeros(2)

    # Target is random
    target = jnp.array([center, center]) + random.normal(k2, (2,)) * 2.0
    target = jnp.clip(target, 1.0, params.world_size - 1.0)

    return EnvState(pos=pos, vel=vel, room_id=room_id, target_pos=target)


# --- 3. DIFFERENTIABLE RENDERER ---


def render(state: EnvState, params: EnvParams) -> jnp.ndarray:
    """
    Generates a 64x64 grayscale image.
    Differentiable w.r.t state.pos (Soft rasterization).
    """
    # Create grid coordinates
    x = jnp.linspace(0, params.world_size, params.img_size)
    y = jnp.linspace(0, params.world_size, params.img_size)
    X, Y = jnp.meshgrid(x, y)

    # --- A. Background Texture (Room Dependent) ---

    # Room A: Grid Pattern (Metal Floor)
    grid_pattern = (jnp.sin(X * 2.0) * jnp.sin(Y * 2.0)) > 0.95
    bg_A = 0.2 + 0.1 * grid_pattern.astype(jnp.float32)

    # Room B: Hazard Stripes (Diagonal)
    stripe_pattern = jnp.sin((X + Y) * 4.0) > 0.0
    bg_B = 0.2 + 0.3 * stripe_pattern.astype(jnp.float32)

    background = jax.lax.select(state.room_id == 0, bg_A, bg_B)

    # --- B. Objects (Soft Circles) ---

    def draw_circle(img, center, radius, intensity):
        # Gaussian falloff for differentiability
        dist_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2
        mask = jnp.exp(-dist_sq / (2 * (radius / 2.0) ** 2))
        return img * (1.0 - mask) + intensity * mask

    # Draw Target (Dark Spot)
    img = draw_circle(background, state.target_pos, 1.0, 0.8)

    # Draw Cargo (Bright Spot)
    img = draw_circle(img, state.pos, 0.8, 1.0)

    return img.reshape(params.img_size, params.img_size, 1)  # (H, W, C)


# --- 4. DEMO ---

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("--- Testing Coriolis Cargo Environment ---")
    key = random.PRNGKey(0)
    params = EnvParams()

    # 1. Room A (Training)
    state_A = reset(key, room_id=0, params=params)
    img_A = render(state_A, params)

    # 2. Room B (Testing)
    state_B = reset(key, room_id=1, params=params)
    img_B = render(state_B, params)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(img_A, cmap="gray", origin="lower")
    ax[0].set_title("Room A: Metal Floor\nGravity: South (Down)")

    ax[1].imshow(img_B, cmap="gray", origin="lower")
    ax[1].set_title("Room B: Hazard Stripes\nGravity: West (Left)")

    plt.savefig("coriolis_env_demo.png")
    print("Environment verified. Saved 'coriolis_env_demo.png'")
