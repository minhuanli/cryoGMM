"""
Farthest-Point Sampling (FPS) utilities for conformational subset construction.

A prior conformational ensemble (an MD/CG trajectory) is usually far too large
to score against cryo-EM images frame by frame. These helpers reduce it to a
compact, representative subset:

  1. Embed every frame in a low-dimensional collective-variable (CV) space.
  2. Pick `n_clusters` maximally spread-out frames with FPS (`fps_clustering`).
  3. Assign every frame to its nearest center, giving the cluster labels the
     GMM build step needs to estimate a local covariance per center.

The on-disk layout produced by `save_cluster_assignment` (`center_idx.npy`,
`cluster_labels.npy`, `center_{k}.pdb`) is exactly what
`scripts/step0_build_gmm/gmm_build.py` expects in its `--cluster_root`.

FPS itself is delegated to the `fpsample` package by default (fast Rust
implementation); a pure-numpy backend is available for environments without it.
"""

import os

import numpy as np


__all__ = [
    "farthest_point_sampling",
    "min_pairwise_d2",
    "refine_centers_max_min",
    "assign_clusters",
    "fps_clustering",
    "resample_cluster_members",
    "save_cluster_assignment",
    "plot_fps_clustering",
]


def farthest_point_sampling(X, n_clusters, start_idx=None, backend="fpsample"):
    """Select `n_clusters` maximally spread-out points with farthest-point sampling.

    Greedy FPS: starting from one point, repeatedly add the point whose distance
    to the closest already-selected point is largest.

    Args:
        X (np.ndarray): Shape (N, D). Points to sample from, typically frames
            embedded in CV space.
        n_clusters (int): Number of centers to select.
        start_idx (int, optional): Index of the first center. If None, it is
            drawn from the global numpy RNG (so `np.random.seed` makes the
            result reproducible for both backends).
        backend (str): "fpsample" (default, requires the `fpsample` package)
            or "numpy" (dependency-free reference implementation).

    Returns:
        np.ndarray: Shape (n_clusters,), int64 frame indices of the centers.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D), got shape {X.shape}")
    n_points = X.shape[0]
    if not 1 <= n_clusters <= n_points:
        raise ValueError(
            f"n_clusters must be in [1, {n_points}], got {n_clusters}"
        )
    if start_idx is None:
        start_idx = int(np.random.randint(low=0, high=n_points))

    if backend == "fpsample":
        try:
            import fpsample
        except ImportError as err:
            raise ImportError(
                "The 'fpsample' package is required for backend='fpsample'. "
                "Install it with `pip install fpsample`, or pass "
                "backend='numpy' to use the built-in implementation."
            ) from err
        centers_idx = fpsample.fps_sampling(X, n_clusters, start_idx=start_idx)
        return np.asarray(centers_idx, dtype=np.int64)

    if backend == "numpy":
        centers_idx = np.empty(n_clusters, dtype=np.int64)
        centers_idx[0] = start_idx
        # Squared distance from every point to the closest selected center
        dist2 = np.sum((X - X[start_idx]) ** 2, axis=1)
        for i in range(1, n_clusters):
            next_idx = int(np.argmax(dist2))
            centers_idx[i] = next_idx
            dist2 = np.minimum(dist2, np.sum((X - X[next_idx]) ** 2, axis=1))
        return centers_idx

    raise ValueError(f"Unknown backend '{backend}', expected 'fpsample' or 'numpy'")


def min_pairwise_d2(points):
    """Smallest squared distance between any two distinct points.

    Args:
        points (np.ndarray): Shape (K, D).

    Returns:
        float: Minimum pairwise squared distance, or `inf` if K < 2.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return np.inf
    d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(d2.min())


def refine_centers_max_min(
    X,
    centers_idx,
    n_iterations=10,
    pool_size=None,
    pool_frac=0.1,
    verbose=False,
):
    """Locally refine FPS centers to increase their minimum pairwise spacing.

    Greedy FPS is order-dependent and can leave two centers closer together than
    necessary. This pass repeatedly proposes replacing one center with a random
    candidate frame and accepts the swap whenever it increases the minimum
    pairwise distance between centers (a max-min / Maximin criterion). It stops
    early once a full sweep produces no improvement.

    Args:
        X (np.ndarray): Shape (N, D). Same CV space used for the FPS selection.
        centers_idx (np.ndarray): Shape (K,). Center indices to refine.
        n_iterations (int): Maximum number of refinement sweeps.
        pool_size (int, optional): Candidate frames drawn per center per sweep.
            Defaults to `pool_frac * N`.
        pool_frac (float): Candidate pool size as a fraction of N, used when
            `pool_size` is None.
        verbose (bool): Print each accepted swap.

    Returns:
        np.ndarray: Shape (K,), int64 refined center indices.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    centers_idx = np.array(centers_idx, dtype=np.int64, copy=True)
    n_points = X.shape[0]
    n_clusters = centers_idx.shape[0]
    if n_clusters < 2:
        return centers_idx
    if pool_size is None:
        pool_size = int(pool_frac * n_points)
    pool_size = int(np.clip(pool_size, 1, n_points))

    centers = X[centers_idx].copy()
    # Pairwise squared distances between centers, kept in sync with `centers`
    d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    best_min_d2 = float(d2.min())

    for it in range(n_iterations):
        improved = False
        # Shuffle the order of centers to avoid biasing early slots
        order = np.arange(n_clusters)
        np.random.shuffle(order)
        for j in order:
            cand_idx = np.random.choice(n_points, pool_size, replace=False)
            # Drop candidates that are already centers, to avoid duplicates
            cand_idx = cand_idx[~np.isin(cand_idx, centers_idx)]
            if cand_idx.size == 0:
                continue
            # Best spacing achievable among the pairs that do not involve j:
            # swapping center j cannot change any of them
            keep = np.ones(n_clusters, dtype=bool)
            keep[j] = False
            base_min_d2 = float(d2[np.ix_(keep, keep)].min()) if n_clusters > 2 else np.inf
            for ci in cand_idx:
                new_row = ((X[ci] - centers) ** 2).sum(axis=1)
                new_row[j] = np.inf
                new_min_d2 = min(base_min_d2, float(new_row.min()))
                if new_min_d2 > best_min_d2 + 1e-12:
                    if verbose:
                        print(
                            f" Iter {it}: center {j} replaced by frame {ci} "
                            f"(min_d2 improved {best_min_d2:.6g} -> {new_min_d2:.6g})",
                            flush=True,
                        )
                    centers[j] = X[ci]
                    centers_idx[j] = int(ci)
                    d2[j, :] = new_row
                    d2[:, j] = new_row
                    best_min_d2 = new_min_d2
                    improved = True
                    break
        if not improved:
            if verbose:
                print("No improvement, stopping early.", flush=True)
            break

    return centers_idx


def assign_clusters(X, centers_idx, chunk_size=100000):
    """Assign every point to its nearest center (Voronoi partition of CV space).

    Args:
        X (np.ndarray): Shape (N, D). Points in CV space.
        centers_idx (np.ndarray): Shape (K,). Indices of the centers into X.
        chunk_size (int): Rows processed per block, to bound peak memory.

    Returns:
        np.ndarray: Shape (N,), int64 cluster label per point, in [0, K).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    centers = X[np.asarray(centers_idx, dtype=np.int64)]
    n_points = X.shape[0]
    labels = np.empty(n_points, dtype=np.int64)
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        d2 = ((X[start:stop, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:stop] = np.argmin(d2, axis=1)
    return labels


def fps_clustering(
    X,
    n_clusters,
    seed=None,
    backend="fpsample",
    refine=False,
    n_iterations=10,
    pool_size=None,
    pool_frac=0.1,
    verbose=False,
):
    """Build a representative conformational subset by FPS in CV space.

    Convenience wrapper around `farthest_point_sampling`, the optional
    `refine_centers_max_min` pass, and `assign_clusters`.

    Args:
        X (np.ndarray): Shape (N_frames, D_cv). Trajectory frames embedded in
            collective-variable space (e.g. two inter-residue distances, or a
            distance and an RMSD to a reference state).
        n_clusters (int): Size of the representative subset.
        seed (int, optional): Seed for the global numpy RNG. FPS is randomised
            only through its starting frame, so a fixed seed makes the whole
            selection reproducible. Run several seeds to get independent
            "sets" and an error bar on the downstream ensemble.
        backend (str): FPS backend, "fpsample" or "numpy".
        refine (bool): Run the max-min refinement pass after FPS.
        n_iterations, pool_size, pool_frac: Passed to `refine_centers_max_min`.
        verbose (bool): Print refinement progress.

    Returns:
        tuple:
            - centers_idx (np.ndarray): Shape (n_clusters,), frame index of
              each representative conformation.
            - labels (np.ndarray): Shape (N_frames,), cluster label per frame.
    """
    if seed is not None:
        np.random.seed(seed)

    centers_idx = farthest_point_sampling(X, n_clusters, backend=backend)
    if refine:
        centers_idx = refine_centers_max_min(
            X,
            centers_idx,
            n_iterations=n_iterations,
            pool_size=pool_size,
            pool_frac=pool_frac,
            verbose=verbose,
        )
    labels = assign_clusters(X, centers_idx)
    return centers_idx, labels


def resample_cluster_members(labels, n_samples=5, seed=None, allow_replacement=False):
    """Draw alternative representatives from within each cluster.

    Used for the resampling variant of the pipeline: instead of running FPS with
    several seeds (which changes the clustering itself), keep one clustering and
    draw `n_samples` independent member frames per cluster. Column i of the
    result is then a complete alternative subset, playing the role of "set i".

    Args:
        labels (np.ndarray): Shape (N_frames,). Cluster label per frame, as
            returned by `assign_clusters`.
        n_samples (int): Frames to draw per cluster.
        seed (int, optional): Seed for the global numpy RNG.
        allow_replacement (bool): Sample with replacement in clusters holding
            fewer than `n_samples` frames instead of raising.

    Returns:
        np.ndarray: Shape (K, n_samples), int64 frame indices. Entry [k, i] is
            the i-th draw from cluster k.
    """
    labels = np.asarray(labels)
    if seed is not None:
        np.random.seed(seed)

    cluster_ids = np.arange(labels.min(), labels.max() + 1)
    members = [np.where(labels == k)[0] for k in cluster_ids]

    if not allow_replacement:
        too_small = [
            (int(k), int(m.size))
            for k, m in zip(cluster_ids, members)
            if m.size < n_samples
        ]
        if too_small:
            raise ValueError(
                f"{len(too_small)} cluster(s) hold fewer than n_samples={n_samples} "
                f"frames: {too_small[:10]}{' ...' if len(too_small) > 10 else ''}. "
                "Use fewer clusters, lower n_samples, or allow_replacement=True."
            )

    return np.array(
        [
            np.random.choice(m, size=n_samples, replace=(m.size < n_samples))
            for m in members
        ],
        dtype=np.int64,
    )


def save_cluster_assignment(
    output_dir,
    centers_idx,
    labels,
    xyz=None,
    topology=None,
    prefix="center",
):
    """Write a cluster assignment in the layout the GMM build step reads.

    Creates `output_dir` and writes:
        center_idx.npy      (K,)  frame index of each representative
        cluster_labels.npy  (N,)  cluster label per frame
        {prefix}_{k}.pdb          one representative conformation per cluster,
                                  when `xyz` and `topology` are given

    Args:
        output_dir (str): Destination directory, e.g.
            `{root}/{n_clusters}_clusters/set_{i}`.
        centers_idx (np.ndarray): Shape (K,). Center frame indices.
        labels (np.ndarray): Shape (N,). Cluster label per frame.
        xyz (np.ndarray, optional): Shape (N, n_atoms, 3). Aligned trajectory
            coordinates the representative PDBs are sliced from.
        topology (mdtraj.Topology, optional): Topology matching `xyz`.
        prefix (str): Basename prefix for the per-center PDB files.
    """
    from cryogmm.utils import save_samples_to_pdb

    os.makedirs(output_dir, exist_ok=True)
    centers_idx = np.asarray(centers_idx, dtype=np.int64)
    np.save(os.path.join(output_dir, "center_idx.npy"), centers_idx)
    np.save(os.path.join(output_dir, "cluster_labels.npy"), np.asarray(labels))

    if xyz is None or topology is None:
        return
    for k, center_id in enumerate(centers_idx):
        save_samples_to_pdb(
            xyz[center_id],
            topology,
            os.path.join(output_dir, f"{prefix}_{k}.pdb"),
        )


def plot_fps_clustering(
    X,
    centers_idx,
    labels=None,
    highlight_idx=None,
    xlabel="CV1",
    ylabel="CV2",
    title=None,
    filename=None,
    show=False,
):
    """Scatter the CV space coloured by cluster, with the centers marked.

    A quick sanity check that the subset covers the populated regions of CV
    space and that no cluster is left nearly empty.

    Args:
        X (np.ndarray): Shape (N, 2). CV embedding; only the first two columns
            are plotted.
        centers_idx (np.ndarray): Shape (K,). Center frame indices.
        labels (np.ndarray, optional): Shape (N,). Colours the points by cluster.
        highlight_idx (np.ndarray, optional): Extra frames to mark, e.g. the
            output of `resample_cluster_members` for one set.
        xlabel, ylabel, title (str): Axis labels and title.
        filename (str, optional): Path to save the figure to.
        show (bool): Call `plt.show()` instead of closing the figure.

    Returns:
        matplotlib.figure.Figure: The figure, for further customisation.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    if X.shape[1] < 2:
        raise ValueError(f"X must have at least 2 columns to plot, got {X.shape[1]}")
    centers_idx = np.asarray(centers_idx, dtype=np.int64)

    fig = plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=1, cmap="tab20")
    plt.scatter(
        X[centers_idx, 0], X[centers_idx, 1],
        c="red", s=15, marker="x", label="FPS centers",
    )
    if highlight_idx is not None:
        highlight_idx = np.asarray(highlight_idx, dtype=np.int64)
        plt.scatter(
            X[highlight_idx, 0], X[highlight_idx, 1],
            c="green", s=30, marker="^", label="Sampled points",
        )
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if title is not None:
        plt.title(title, fontsize=14)
    plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
