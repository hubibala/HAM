"""
experiment_h4_simulation.py
===================================
H4 — Forward Predictive Simulation using Exponential Map and ODEs

We compare 4 distinct predictive models:
1. Null (Riemannian): Geometry only, no directional wind.
2. Ablation (Latent ODE): Directional wind only, ignoring curvature.
3. Randers (HAM): Geometry + Wind interacting together.
4. SOTA Baseline (PCA ODE): Neural ODE on raw PCA space.

Evaluated using Minimum Point-to-Trajectory Distance against true future cell states.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import anndata

from ham.bio.data import BioDataset
from ham.solvers.geodesic import ExponentialMap

# Reuse helpers
from weinreb_vae import build_diagnostic_vae, encode_all, TARGET_FATES
from weinreb_experiment import attach_datadriven_randers_metric, load_phase1_vae
from experiment_h3_discriminative import build_riemannian_fallback

# ── 1. Utilities for ODEs ────────────────────────────────────────────────────

class PCANeuralODE(eqx.Module):
    mlp: eqx.nn.MLP
    
    def __init__(self, data_dim: int, key: jax.random.PRNGKey):
        self.mlp = eqx.nn.MLP(data_dim, data_dim, width_size=128, depth=3, activation=jax.nn.silu, key=key)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.mlp(x)

def trace_ode(vector_field_fn, z0: jax.Array, step_size: float = 0.01, max_steps: int = 400) -> jax.Array:
    """Standard RK4 Integrator for Neural ODEs"""
    def scan_fn(z, _):
        k1 = vector_field_fn(z)
        k2 = vector_field_fn(z + step_size/2 * k1)
        k3 = vector_field_fn(z + step_size/2 * k2)
        k4 = vector_field_fn(z + step_size * k3)
        z_next = z + step_size/6 * (k1 + 2*k2 + 2*k3 + k4)
        return z_next, z_next
    _, traj = jax.lax.scan(scan_fn, z0, jnp.arange(max_steps))
    return jnp.concatenate([z0[None], traj], axis=0)


def train_pca_ode(dataset: BioDataset, key: jax.random.PRNGKey, epochs: int = 15, batch_size: int = 256) -> PCANeuralODE:
    """Trains a simple dx/dt = f(x) model on PCA space"""
    print("  Training PCA-space Neural ODE (SOTA Baseline) ...")
    model = PCANeuralODE(dataset.X.shape[1], key)
    
    v_norms = jnp.linalg.norm(dataset.V, axis=1)
    mask = v_norms > 1e-6
    X_train = dataset.X[mask]
    V_train = dataset.V[mask]
    
    optimizer = optax.adamw(1e-3)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    @eqx.filter_jit
    def step(model, opt_state, x, v):
        def loss_fn(m):
            v_pred = jax.vmap(m)(x)
            return jnp.mean((v_pred - v)**2)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    for ep in range(epochs):
        np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            idx = indices[i:i+batch_size]
            model, opt_state, _ = step(model, opt_state, X_train[idx], V_train[idx])
    return model

def compute_trajectory_metrics(traj: jax.Array, target_z: jax.Array, centroids: dict, target_fate: str) -> tuple[float, int]:
    """
    Computes minimum point-to-trajectory distance to the actual target cell,
    and returns 1 if the closest point on the trajectory correctly classifies into the true fate.
    """
    # Evaluate strictly based on the FINAL point of the trajectory
    end_z = traj[-1]
    
    dist_to_true = float(jnp.linalg.norm(end_z - target_z))
    
    d_true_cent = jnp.linalg.norm(end_z - centroids[target_fate])
    d_cf_cent = min(float(jnp.linalg.norm(end_z - c)) for f, c in centroids.items() if f != target_fate)
    
    hit = 1 if d_true_cent < d_cf_cent else 0
    return dist_to_true, hit

# ── 2. Main ──────────────────────────────────────────────────────────────

def main():
    CHECKPOINT   = "data/weinreb_vae_phase1.eqx"
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    LATENT_DIM   = 8
    N_EVAL       = 1000

    print("=" * 70)
    print("H4 — FORWARD SIMULATION (Trajectory Evaluation)")
    print("Comparing Null, Latent ODE, Randers, and PCA SOTA Neural ODE")
    print("=" * 70)

    # Load data
    adata = anndata.read_h5ad(PREPROCESSED)
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    V_pca = np.array(adata.obsm['velocity_pca'], dtype=np.float32)
    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)

    scaler = joblib.load("data/weinreb_scaler.joblib")
    X_norm = scaler.transform(X_pca).astype(np.float32)
    V_norm = (V_pca / (scaler.scale_ + 1e-8)).astype(np.float32)

    dataset = BioDataset(X=jnp.array(X_norm), V=jnp.array(V_norm), labels=jnp.array(labels), lineage_pairs=None)
    test_triples = np.load(TEST_TRIPLES)[:N_EVAL]

    # Models
    key = jax.random.PRNGKey(42)
    vae_p1 = load_phase1_vae(CHECKPOINT, X_norm.shape[1], LATENT_DIM, len(fate_names), key)
    vae_randers = attach_datadriven_randers_metric(vae_p1, dataset, n_anchors=2000, sigma=0.4)
    vae_null = build_riemannian_fallback(vae_randers)
    pca_ode = train_pca_ode(dataset, jax.random.PRNGKey(7))

    Z_all = encode_all(vae_randers, jnp.array(X_norm))

    # Fate centroids
    all_triples = np.load(TEST_TRIPLES)
    day6_indices = np.unique(all_triples[:, 2])
    day6_label_mask = np.zeros(len(labels), dtype=bool)
    day6_label_mask[day6_indices] = True
    
    fate_centroids = {}
    for fname in TARGET_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        mask = (labels == fidx) & day6_label_mask
        if mask.sum() > 10:
            fate_centroids[fname] = jnp.array(Z_all[mask].mean(axis=0))

    # Prepare starts
    idx_day2 = test_triples[:, 0]
    idx_day6 = test_triples[:, 2]
    
    true_fate_names = [fate_names[labels[i]] for i in idx_day6]
    valid_mask = np.array([f in fate_centroids for f in true_fate_names])
    
    idx_day2_v = idx_day2[valid_mask]
    idx_day6_v = idx_day6[valid_mask]
    true_fates_v = [true_fate_names[i] for i in np.where(valid_mask)[0]]

    # Map to Latent
    print(f"Evaluating {len(idx_day2_v)} valid day2->day6 trajectories.")
    
    # Batch inputs
    X_start_v = jnp.array(X_norm[idx_day2_v])
    V_start_v = jnp.array(V_norm[idx_day2_v])
    Z_start_v = Z_all[idx_day2_v]
    Z_end_v   = Z_all[idx_day6_v]
    
    # Project Initial velocity to Latent Space
    _, V_lat_v = jax.vmap(vae_randers.project_control)(X_start_v, V_start_v)

    shooter = ExponentialMap(step_size=0.01, max_steps=400)

    # ── Simulation Fns ────────────────────────────────────────────────────────
    
    def get_w(z_pt):
        return vae_randers.metric._get_zermelo_data(z_pt)[1]
    
    @eqx.filter_jit
    @eqx.filter_vmap
    def sim_riemannian(z):
        traj, _ = shooter.trace(vae_null.metric, z, get_w(z), t_max=4.0)
        return traj

    @eqx.filter_jit
    @eqx.filter_vmap
    def sim_randers(z):
        traj, _ = shooter.trace(vae_randers.metric, z, get_w(z), t_max=4.0)
        return traj

    @eqx.filter_jit
    @eqx.filter_vmap
    def sim_latent_ode(z):
        return trace_ode(get_w, z)

    @eqx.filter_jit
    @eqx.filter_vmap
    def sim_pca_ode(x):
        traj_pca = trace_ode(pca_ode, x)
        return jax.vmap(vae_randers._get_dist)(traj_pca).mean

    models = {
        "Null (Riemannian)": sim_riemannian,
        "Ablation (Latent ODE)": sim_latent_ode,
        "Randers (HAM)": sim_randers,
        "SOTA (PCA ODE)": sim_pca_ode,
    }

    print("\n" + "="*80)
    print("5-FOLD BOOTSTRAP EVALUATION (Mean ± Std over 5 random subsets of 150 cells)")
    print(f"{'Model':>25} {'Endpoint Dist':>20} {'Endpoint Hit':>20}")
    print("-" * 80)

    n_folds = 5
    fold_size = min(150, len(idx_day2_v))
    
    rng = np.random.default_rng(42)
    fold_indices = [rng.choice(len(idx_day2_v), fold_size, replace=False) for _ in range(n_folds)]

    results = {}
    
    import time
    for name, sim_fn in models.items():
        t0 = time.time()
        fold_dists = []
        fold_hits = []
        
        for fold in fold_indices:
            if "PCA" in name:
                traj_batch = sim_fn(X_start_v[fold])
            else:
                traj_batch = sim_fn(Z_start_v[fold])

            dists = []
            hits = 0
            for i, traj in enumerate(traj_batch):
                d, h = compute_trajectory_metrics(traj, Z_end_v[fold[i]], fate_centroids, true_fates_v[fold[i]])
                dists.append(d)
                hits += h
                
            fold_dists.append(np.median(dists))
            fold_hits.append(hits / len(fold))
            
        t1 = time.time()
        
        mean_dist = np.mean(fold_dists)
        std_dist = np.std(fold_dists)
        mean_hit = np.mean(fold_hits)
        std_hit = np.std(fold_hits)
        
        results[name] = {"dist": mean_dist, "hit": mean_hit, "time": t1-t0}
        
        print(f"  {name:>23s} -> {mean_dist:8.3f} ± {std_dist:5.3f}    {mean_hit:6.1%} ± {std_hit:5.1%}   ({t1-t0:.1f}s/fold)")

    print("\nH4 (Trajectory Distance Evaluation) completed.")
    
    # ── Visualization ────────────────────────────────────────────────────────
    print("\nGenerating trajectory visualization...")
    valid_idx = np.all(np.isfinite(Z_all), axis=1)
    Z_clean = Z_all[valid_idx]
    pca2 = PCA(n_components=2).fit(Z_clean)
    z2d_all = pca2.transform(Z_all)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    _all_colors = {"Monocyte": "steelblue", "Neutrophil": "tomato", "Erythroid": "forestgreen", "Megakaryocyte": "darkorange"}
    colors_fates = {f: _all_colors.get(f, "black") for f in TARGET_FATES if f in fate_centroids}
    
    for ax, (name, sim_fn) in zip(axes, models.items()):
        ax.scatter(z2d_all[:, 0], z2d_all[:, 1], s=1, alpha=0.1, color='lightgray')
        for fname, centroid in fate_centroids.items():
            c2d = pca2.transform(np.array(centroid)[None])[0]
            color = colors_fates.get(fname, "black")
            ax.scatter(*c2d, s=150, marker='X', color=color, edgecolor='white', zorder=10)
        
        ax.set_title(f"{name}\nDist: {results[name]['dist']:.3f} | Hit: {results[name]['hit']:.1%}")
        
        N_VIZ = 50
        for fname, color in colors_fates.items():
            potential_indices = [i for i, f in enumerate(true_fates_v) if f == fname]
            for i, idx_v in enumerate(potential_indices[:N_VIZ]):
                if "PCA" in name:
                    traj = sim_fn(X_start_v[idx_v:idx_v+1])[0]
                else:
                    traj = sim_fn(Z_start_v[idx_v:idx_v+1])[0]
                
                t2d = pca2.transform(np.array(traj))
                true_end_2d = pca2.transform(np.array(Z_end_v[idx_v:idx_v+1]))[0]
                
                ax.plot(t2d[:, 0], t2d[:, 1], color=color, lw=1.0, alpha=0.3, zorder=5)
                ax.scatter(t2d[0, 0], t2d[0, 1], color=color, s=10, marker='o', edgecolors='none', alpha=0.4, zorder=6)
                ax.scatter(t2d[-1, 0], t2d[-1, 1], color=color, s=40, marker='*', edgecolors='black', alpha=0.9, zorder=7)
                
                # Plot the actual target cell and connect it to the predicted endpoint
                ax.scatter(true_end_2d[0], true_end_2d[1], color=color, s=30, marker='d', edgecolors='black', alpha=0.7, zorder=8)
                ax.plot([t2d[-1, 0], true_end_2d[0]], [t2d[-1, 1], true_end_2d[1]], color='gray', linestyle='dotted', lw=0.8, alpha=0.5, zorder=4)

    plt.tight_layout()
    plt.savefig("h4_fate_trajectories.png", dpi=200, bbox_inches='tight')
    print(f"Saved visualization -> h4_fate_trajectories.png")

if __name__ == "__main__":
    main()