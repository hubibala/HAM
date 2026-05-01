"""
experiment_baseline_cellrank.py
===============================
SOTA Baseline Comparison using scVelo / CellRank.

Computes the scVelo transition matrix on the 50D PCA space and simulates
a directed random walk to evaluate trajectory prediction on the same 5-fold
cross-validation sets used in H4.
"""

import time
import numpy as np
import jax.numpy as jnp
import jax
import anndata
import joblib
from sklearn.decomposition import PCA

import scvelo as scv
from cellrank.kernels import VelocityKernel

from weinreb_vae import TARGET_FATES
from experiment_h4_simulation import compute_trajectory_metrics

def main():
    PREPROCESSED = "data/weinreb_preprocessed.h5ad"
    TEST_TRIPLES = "data/weinreb_test_triples.npy"
    N_EVAL = 1000

    print("=" * 80)
    print("SOTA BASELINE — CellRank / scVelo (Velocity-Directed Random Walk)")
    print("Evaluating on the exact same 5-fold cross-validation triples as H4.")
    print("=" * 80)

    # 1. Load Data
    t0 = time.time()
    adata = anndata.read_h5ad(PREPROCESSED)
    labels = adata.obs['Cell type annotation'].cat.codes.values.astype(np.int32)
    fate_names = list(adata.obs['Cell type annotation'].cat.categories)
    
    print(f"Loaded {len(adata)} cells in {time.time()-t0:.1f}s")

    # 2. Build CellRank Kernel
    print("\n--- Training Phase ---")
    
    t0 = time.time()
    print("Computing scVelo KNN Graph...")
    scv.pp.neighbors(adata, n_pcs=50, use_rep='X_pca')
    knn_time = time.time() - t0
    print(f"  -> Done in {knn_time:.1f}s")
    
    t0 = time.time()
    print("Computing CellRank/scVelo Transition Matrix...")
    scv.tl.velocity_graph(adata, vkey='velocity_pca', xkey='X_pca')
    P = scv.utils.get_transition_matrix(adata)
    transition_time = time.time() - t0
    print(f"  -> Done in {transition_time:.1f}s")
    
    total_train_time = knn_time + transition_time
    print(f"Total SOTA Training Time: {total_train_time:.1f}s")

    # 3. Prepare Evaluation Sets (Exact same seed/logic as H4)
    print("\n--- Evaluation Phase ---")
    test_triples = np.load(TEST_TRIPLES)[:N_EVAL]
    
    # We need fate centroids in PCA space
    X_pca = np.array(adata.obsm['X_pca'], dtype=np.float32)
    
    # Recreate the valid triples logic from H4
    all_triples = np.load(TEST_TRIPLES)
    day6_indices = np.unique(all_triples[:, 2])
    day6_label_mask = np.zeros(len(labels), dtype=bool)
    day6_label_mask[day6_indices] = True
    
    fate_centroids_pca = {}
    for fname in TARGET_FATES:
        if fname not in fate_names: continue
        fidx = fate_names.index(fname)
        mask = (labels == fidx) & day6_label_mask
        if mask.sum() > 10:
            fate_centroids_pca[fname] = jnp.array(X_pca[mask].mean(axis=0))

    idx_day2 = test_triples[:, 0]
    idx_day6 = test_triples[:, 2]
    
    true_fate_names = [fate_names[labels[i]] for i in idx_day6]
    valid_mask = np.array([f in fate_centroids_pca for f in true_fate_names])
    
    idx_day2_v = idx_day2[valid_mask]
    idx_day6_v = idx_day6[valid_mask]
    true_fates_v = [true_fate_names[i] for i in np.where(valid_mask)[0]]

    # 5-Fold setup
    n_folds = 5
    fold_size = min(150, len(idx_day2_v))
    rng = np.random.default_rng(42)
    fold_indices = [rng.choice(len(idx_day2_v), fold_size, replace=False) for _ in range(n_folds)]

    print(f"Evaluating 5 random folds of {fold_size} day2->day6 trajectories.")
    
    # 4. Simulate Random Walk
    # P is scipy.sparse.csr_matrix. We can propagate probabilities.
    def simulate_random_walk(start_idx, n_steps=60):
        # Initial distribution
        p = np.zeros(P.shape[0])
        p[start_idx] = 1.0
        
        traj_pca = [X_pca[start_idx]]
        
        # Power iteration
        for _ in range(n_steps):
            p = P.T.dot(p)  # Walk forward
            expected_pos = p.dot(X_pca)
            traj_pca.append(expected_pos)
            
        return jnp.array(traj_pca)

    fold_dists = []
    fold_hits = []
    
    t0_eval = time.time()
    for fold in fold_indices:
        dists = []
        hits = 0
        for i in fold:
            start_cell = idx_day2_v[i]
            target_cell_pca = jnp.array(X_pca[idx_day6_v[i]])
            true_fate = true_fates_v[i]
            
            # Predict trajectory via random walk expected positions
            traj = simulate_random_walk(start_cell, n_steps=60)
            
            d, h = compute_trajectory_metrics(traj, target_cell_pca, fate_centroids_pca, true_fate)
            dists.append(d)
            hits += h
            
        fold_dists.append(np.median(dists))
        fold_hits.append(hits / len(fold))

    t_eval = time.time() - t0_eval
    
    mean_dist = np.mean(fold_dists)
    std_dist = np.std(fold_dists)
    mean_hit = np.mean(fold_hits)
    std_hit = np.std(fold_hits)
    
    print("\n" + "="*80)
    print("5-FOLD BOOTSTRAP RESULTS (CellRank / scVelo)")
    print("-" * 80)
    print(f"  {'CellRank Random Walk':>23s} -> {mean_dist:8.3f} ± {std_dist:5.3f}    {mean_hit:6.1%} ± {std_hit:5.1%}   ({t_eval/5:.1f}s/fold)")
    print("=" * 80)

if __name__ == "__main__":
    main()
