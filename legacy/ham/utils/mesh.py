import numpy as np
import jax.numpy as jnp
from typing import Tuple


def generate_icosphere(
    subdivisions: int = 3, radius: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generates a geodesic polyhedron (icosphere) approximating a unit sphere.

    Args:
        subdivisions: Number of recursive subdivisions (0 = Icosahedron).
        radius: Scale of the sphere.

    Returns:
        vertices: (V, 3) JAX array of vertex positions.
        faces: (F, 3) JAX array of vertex indices (triangles).
    """
    # 1. Base Icosahedron (Golden Ratio)
    t = (1.0 + np.sqrt(5.0)) / 2.0

    # Vertices (12)
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float32,
    )

    # Normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1)[:, None]

    # Faces (20)
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # 2. Subdivision Loop
    for _ in range(subdivisions):
        midpoint_cache = {}
        new_faces = []

        def get_midpoint(v1_idx, v2_idx):
            # FIX: Nonlocal declaration must be the FIRST line
            nonlocal verts

            # Sort indices to ensure uniqueness of edge key
            key = tuple(sorted((v1_idx, v2_idx)))

            if key not in midpoint_cache:
                # Compute midpoint
                mid = (verts[v1_idx] + verts[v2_idx]) / 2
                mid /= np.linalg.norm(mid)  # Project to sphere

                # Add to vertices
                midpoint_cache[key] = len(verts)
                verts = np.vstack([verts, mid])

            return midpoint_cache[key]

        for i in range(len(faces)):
            v0, v1, v2 = faces[i]

            # Get midpoints of edges
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)

            # Create 4 new faces
            new_faces.append([v0, a, c])
            new_faces.append([v1, b, a])
            new_faces.append([v2, c, b])
            new_faces.append([a, b, c])

        faces = np.array(new_faces, dtype=np.int32)

    # 3. Scale and return as numpy arrays
    verts *= radius

    # Convert to JAX arrays before returning
    return jnp.array(verts), jnp.array(faces)
