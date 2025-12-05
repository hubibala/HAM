import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from ham.geometry import Sphere, Randers
from ham.models import NeuralRanders
from ham.solvers import AVBDSolver
from ham.vis import setup_3d_plot, plot_vector_field, plot_sphere

print("--- HAMTools Phase 3: Physics Discovery (Corrected) ---")

# 1. Ground Truth World
print("Generating Ground Truth Data...")
manifold = Sphere(radius=1.0)

def true_w_net(x): return 0.7 * jnp.array([-x[1], x[0], 0.0])
def true_h_net(x): return jnp.eye(3)

gt_metric = Randers(manifold, true_h_net, true_w_net)
solver = AVBDSolver(step_size=0.05, max_iter=200)

# Generate richer dataset (64 paths to cover sphere better)
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
starts = manifold.random_sample(k1, (64,))
ends   = manifold.random_sample(k2, (64,))

n_steps = 20
dt = 1.0 / n_steps  # The critical time step scaling

trajectories = []
for s, e in zip(starts, ends):
    traj = solver.solve(gt_metric, s, e, n_steps=n_steps)
    trajectories.append(traj)

data_x, data_v, data_a = [], [], []
for traj in trajectories:
    xs = traj.xs
    
    # Finite Differences
    delta_x = xs[1:] - xs[:-1]
    delta_v = delta_x[1:] - delta_x[:-1]
    
    # SCALING FIX: Convert deltas to physical derivatives
    v_phys = delta_x[:-1] / dt
    a_phys = delta_v / (dt ** 2)
    
    data_x.append(xs[1:-1])
    data_v.append(v_phys)
    data_a.append(a_phys)

X = jnp.concatenate(data_x)
V = jnp.concatenate(data_v)
A_target = jnp.concatenate(data_a)

print(f"Dataset Size: {len(X)} points")
print(f"Mean Target Accel Norm: {jnp.mean(jnp.linalg.norm(A_target, axis=1)):.4f}")

# 2. Neural Model
print("Initializing Neural Randers Model...")
model = NeuralRanders(manifold, key=jax.random.PRNGKey(42), hidden_dim=32)
# Lower learning rate for stability with unscaled targets
optim = optax.adam(learning_rate=1e-3)
opt_state = optim.init(eqx.filter(model, eqx.is_array))

# 3. Training
@eqx.filter_value_and_grad
def compute_loss(model, x, v, a_target):
    a_pred = jax.vmap(model.geod_acceleration)(x, v)
    return jnp.mean((a_pred - a_target)**2)

print("Training (Inverse Dynamics)...")
losses = []
for i in range(101):
    loss, grads = compute_loss(model, X, V, A_target)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    losses.append(loss)
    if i % 10 == 0: print(f"Epoch {i}: Loss = {loss:.6f}")

# 4. Visualization
print("\nVisualizing Results...")
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(losses)
ax1.set_title("Inverse Dynamics Loss")
ax1.set_yscale('log')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_sphere(ax2, alpha=0.05)

# Sample equator
theta = np.linspace(0, 2*np.pi, 20)
pts = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)

# True Wind
true_vecs = np.array(jax.vmap(true_w_net)(jnp.array(pts)))

# Learned Effective Wind (Internal)
def get_effective_wind(model, x):
    _, W, _ = model._get_zermelo_data(x)
    return W
    
learned_vecs = np.array(jax.vmap(lambda x: get_effective_wind(model, x))(jnp.array(pts)))

plot_vector_field(ax2, pts, true_vecs, color='blue', alpha=0.3, scale=0.2, label='True')
plot_vector_field(ax2, pts, learned_vecs, color='red', alpha=1.0, scale=0.2, label='Learned')

ax2.set_title("Physics Discovery: Corrected Scaling")
ax2.legend()
plt.savefig("physics_discovery_corrected.png")
plt.show()