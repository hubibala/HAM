import jax
import jax.numpy as jnp
from jax import vmap, lax
from .geodesic import ProjectedGradientSolver


class MultiPathSolver:
    """
    A Global Geodesic Solver.
    Initializes multiple candidate paths (hypotheses) covering different
    topologies (e.g., Short way, Long way, Over poles).
    Converges all of them and selects the one with the Global Minimum Energy.
    """

    def __init__(self, manifold, lr=0.01, max_iters=500, num_candidates=4):
        self.inner_solver = ProjectedGradientSolver(manifold, lr, max_iters)
        self.manifold = manifold
        self.num_candidates = num_candidates

    def _generate_hypotheses(self, start, end, steps):
        """
        Generates K initial path guesses covering the sphere.
        """
        # 1. Linear (Short Path)
        t = jnp.linspace(0, 1, steps)[1:-1]
        base = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]

        # 2. Define Perturbations (The "Ghosts")
        # We add sine waves to pull the path to different hemispheres
        # Midpoint of base path
        mid = (start + end) / 2
        # Orthogonal directions
        north = jnp.array([0.0, 0.0, 1.0])
        south = jnp.array([0.0, 0.0, -1.0])
        back = -mid  # Push away from center

        # Create candidates
        # Path 0: Direct
        p0 = base
        # Path 1: North Arch
        p1 = base + north[None, :] * 0.8 * jnp.sin(t[:, None] * jnp.pi)
        # Path 2: South Arch
        p2 = base + south[None, :] * 0.8 * jnp.sin(t[:, None] * jnp.pi)
        # Path 3: Long Way (Push through center to back)
        p3 = base + back[None, :] * 2.0 * jnp.sin(t[:, None] * jnp.pi)

        # Stack and Project
        candidates = jnp.stack([p0, p1, p2, p3])
        # Project all points to manifold
        return vmap(vmap(self.manifold.projection))(candidates)

    def solve(self, energy_fn, start, end, dummy_init_path):
        """
        Args:
            dummy_init_path: Used only to determine 'steps' (shape).
        """
        steps = dummy_init_path.shape[0] + 2

        # 1. Generate Candidates
        init_paths = self._generate_hypotheses(start, end, steps)

        # 2. Solve All in Parallel (vmap)
        # solve signature: (energy_fn, start, end, init)
        # We vmap over the 'init' argument (axis 0)
        solve_parallel = vmap(self.inner_solver.solve, in_axes=(None, None, None, 0))

        final_paths = solve_parallel(energy_fn, start, end, init_paths)

        # 3. Evaluate Energies
        # energy_fn takes full path (steps, dim)
        energies = vmap(energy_fn)(final_paths)

        # 4. Pick Winner (Argmin)
        best_idx = jnp.argmin(energies)
        best_path = final_paths[best_idx]

        return best_path
