"""
Conformational subset construction — farthest-point sampling (FPS) in CV space

This is the first step of the pipeline, run once on the prior ensemble before
any likelihood or reweighting calculation. It reduces a large CG/MD trajectory
to a compact, representative subset of conformations, plus the cluster
assignment the GMM build step needs to estimate a local covariance per center.

Algorithm:
  - Load the trajectory and superpose every frame onto frame 0
  - Embed each frame in a low-dimensional collective-variable (CV) space
  - Select n_clusters maximally spread-out frames with FPS
  - Assign every frame to its nearest center (Voronoi partition of CV space)
  - Repeat for several seeds to get independent "sets"

Two modes:
  seeds      (default) One independent FPS clustering per seed. Sets differ in
             both which frames are centers and how frames are grouped.
  resample   One shared clustering, then several alternative representative
             frames drawn from within each cluster. Sets differ only in which
             member represents each cluster.

Output structure (seeds mode, consumed by scripts/step0_build_gmm/gmm_build.py
as --cluster_root {output_root}/{n_clusters}_clusters):
  {output_root}/
    {n_clusters}_clusters/
      set_{i}/
        center_idx.npy       # (K,)  frame index of each representative
        cluster_labels.npy   # (N,)  cluster label per frame
        center_{k}.pdb       # representative conformations
        clustering.png       # CV-space diagnostic plot

Output structure (resample mode, use gmm_build.py --shared_clusters):
  {output_root}/
    {n_clusters}_clusters/
      center_idx.npy         # shared clustering
      cluster_labels.npy
      set_{i}/
        sample_{k}.pdb       # alternative representative for cluster k
        sample_idx.txt       # frame index of each
        sampling.png
"""

import os
import argparse
from pathlib import Path

import numpy as np

from cryogmm.preprocess import (
    dist,
    angle,
    torsion,
    fps_clustering,
    resample_cluster_members,
    save_cluster_assignment,
    plot_fps_clustering,
)
from cryogmm.utils import load_aligned_trajectory, save_samples_to_pdb


# ============================================================================
# COLLECTIVE VARIABLES
# ============================================================================
def compute_cv(spec, xyz):
    """Evaluate one --cv spec against aligned coordinates of shape (N, n_atoms, 3).

    Supported specs:
      dist:i,j          distance between atoms i and j
      angle:i,j,k       angle at atom j
      torsion:i,j,k,l   dihedral angle
      file:path         one column of text, loaded with np.loadtxt (e.g. an
                        externally computed RMSD to a reference state)
      npy:path          a 1-D .npy array
    """
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()

    if kind == "file":
        values = np.loadtxt(rest)
    elif kind == "npy":
        values = np.load(rest)
    else:
        try:
            idx = [int(x) for x in rest.split(",")]
        except ValueError:
            raise ValueError(f"Could not parse atom indices in --cv '{spec}'")
        fn, n_expected = {
            "dist": (dist, 2),
            "angle": (angle, 3),
            "torsion": (torsion, 4),
        }.get(kind, (None, 0))
        if fn is None:
            raise ValueError(
                f"Unknown CV type '{kind}' in --cv '{spec}'. Expected one of "
                "dist, angle, torsion, file, npy."
            )
        if len(idx) != n_expected:
            raise ValueError(
                f"--cv '{spec}': {kind} needs {n_expected} atom indices, got {len(idx)}"
            )
        values = fn(*[xyz[:, i] for i in idx])

    values = np.asarray(values).squeeze()
    if values.ndim != 1:
        raise ValueError(f"--cv '{spec}' produced shape {values.shape}, expected 1-D")
    if values.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"--cv '{spec}' has {values.shape[0]} values but the trajectory has "
            f"{xyz.shape[0]} frames"
        )
    return values


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FPS conformational subset construction in collective-variable space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python fps_clustering.py \\\n"
            "      --traj_path positions_all_traj.pt --traj_top top.pdb \\\n"
            "      --alignment_selection '(resi > 108) and name BB2' \\\n"
            "      --cv dist:325,785 --cv dist:33,529 \\\n"
            "      --cv_labels 'CV1, dist(A50-C120)' 'CV2, dist(G6-A81)' \\\n"
            "      --n_clusters 40 --output_root ./fps_clustering\n"
        ),
    )

    # --- paths ---
    parser.add_argument(
        "--traj_path", type=str, required=True,
        help="Trajectory coordinates: a .pt tensor of shape (N_frames, N_atoms, 3) "
             "in nm, or any MDTraj-readable trajectory file.",
    )
    parser.add_argument(
        "--traj_top", type=str, required=True,
        help="Path to topology PDB file.",
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Output root directory. Results go to {output_root}/{n_clusters}_clusters/.",
    )
    parser.add_argument(
        "--alignment_selection", type=str, default=None,
        help="MDTraj selection string for superposition onto frame 0 "
             "(e.g. '(resi > 108) and name BB2'). Omit to skip alignment.",
    )

    # --- collective variables ---
    parser.add_argument(
        "--cv", type=str, action="append", default=None, metavar="SPEC",
        help="A collective variable, repeatable (one --cv per dimension). "
             "Formats: dist:i,j | angle:i,j,k | torsion:i,j,k,l | file:path | npy:path. "
             "Atom indices refer to the aligned trajectory.",
    )
    parser.add_argument(
        "--cv_path", type=str, default=None,
        help="Precomputed CV array (.npy) of shape (N_frames, D), used instead of --cv.",
    )
    parser.add_argument(
        "--cv_labels", type=str, nargs="*", default=None,
        help="Axis labels for the diagnostic plot, one per CV.",
    )

    # --- clustering parameters ---
    parser.add_argument("--n_clusters", type=int, default=40,
                        help="Size of the representative subset (default: 40).")
    parser.add_argument("--seeds", type=str, default="42,12345,162,160,70",
                        help="Comma-separated RNG seeds, one per set "
                             "(default: 42,12345,162,160,70).")
    parser.add_argument("--backend", type=str, default="fpsample",
                        choices=["fpsample", "numpy"],
                        help="FPS implementation (default: fpsample).")
    parser.add_argument("--angstrom_to_nm", action="store_true",
                        help="Divide trajectory coordinates by 10 (for Angstrom-stored "
                             "trajectories).")

    # --- optional max-min refinement ---
    parser.add_argument("--refine", action="store_true",
                        help="Run the max-min refinement pass after FPS. Slow; only "
                             "worth it for the numpy backend or small trajectories.")
    parser.add_argument("--n_iterations", type=int, default=10,
                        help="Maximum refinement sweeps (default: 10).")
    parser.add_argument("--pool_frac", type=float, default=0.1,
                        help="Refinement candidate pool, as a fraction of N_frames "
                             "(default: 0.1).")

    # --- mode ---
    parser.add_argument("--mode", type=str, default="seeds",
                        choices=["seeds", "resample"],
                        help="'seeds': one independent clustering per seed. "
                             "'resample': one shared clustering, alternative "
                             "representatives per set (default: seeds).")
    parser.add_argument("--n_sets", type=int, default=5,
                        help="Number of sets in resample mode (default: 5).")
    parser.add_argument("--resample_seed", type=int, default=20250917,
                        help="RNG seed for the member draw in resample mode.")
    parser.add_argument("--no_pdb", action="store_true",
                        help="Skip writing the representative PDB files.")

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    args = parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    n_clusters = args.n_clusters
    cluster_root = Path(args.output_root) / f"{n_clusters}_clusters"

    print("Run config:", flush=True)
    print(f"  mode        : {args.mode}", flush=True)
    print(f"  n_clusters  : {n_clusters}", flush=True)
    print(f"  backend     : {args.backend}", flush=True)
    print(f"  seeds       : {seeds}", flush=True)
    print(f"  cluster_root: {cluster_root}", flush=True)

    # ------------------------------------------------------------------
    # Load and align trajectory
    # ------------------------------------------------------------------
    print("\nLoading and aligning trajectory...", flush=True)
    traj = load_aligned_trajectory(
        args.traj_path,
        args.traj_top,
        alignment_selection=args.alignment_selection,
        angstrom_to_nm=args.angstrom_to_nm,
    )
    xyz = traj.xyz
    print(f"Trajectory: {xyz.shape[0]} frames, {xyz.shape[1]} atoms", flush=True)

    # ------------------------------------------------------------------
    # Build the CV embedding
    # ------------------------------------------------------------------
    if args.cv_path is not None:
        X = np.load(args.cv_path)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"--cv_path has {X.shape[0]} rows but the trajectory has "
                f"{xyz.shape[0]} frames"
            )
        print(f"Loaded CV embedding from {args.cv_path}: shape {X.shape}", flush=True)
    else:
        if not args.cv:
            raise ValueError("Provide at least one --cv spec, or --cv_path.")
        print(f"Computing {len(args.cv)} collective variable(s)...", flush=True)
        X = np.vstack([compute_cv(spec, xyz) for spec in args.cv]).T
        for spec, col in zip(args.cv, X.T):
            print(f"  {spec:<24} range [{col.min():.3f}, {col.max():.3f}]", flush=True)

    cv_labels = args.cv_labels or [f"CV{i + 1}" for i in range(X.shape[1])]
    if len(cv_labels) < 2:
        cv_labels = list(cv_labels) + ["CV2"]

    # ------------------------------------------------------------------
    # Mode: seeds — one independent FPS clustering per seed
    # ------------------------------------------------------------------
    if args.mode == "seeds":
        for i, seed in enumerate(seeds):
            print(f"\n{'=' * 70}", flush=True)
            print(f"Set {i} (seed {seed})", flush=True)
            print(f"{'=' * 70}", flush=True)

            centers_idx, labels = fps_clustering(
                X, n_clusters,
                seed=seed,
                backend=args.backend,
                refine=args.refine,
                n_iterations=args.n_iterations,
                pool_frac=args.pool_frac,
                verbose=True,
            )
            counts = np.bincount(labels, minlength=n_clusters)
            print(f"Cluster occupancy: min {counts.min()}, median "
                  f"{int(np.median(counts))}, max {counts.max()}", flush=True)
            if counts.min() < 10:
                print(f"  WARNING: {(counts < 10).sum()} cluster(s) hold fewer than 10 "
                      "frames; their covariance will be poorly determined.", flush=True)

            output_dir = cluster_root / f"set_{i}"
            save_cluster_assignment(
                output_dir, centers_idx, labels,
                xyz=None if args.no_pdb else xyz,
                topology=None if args.no_pdb else traj.top,
            )
            plot_fps_clustering(
                X, centers_idx, labels,
                xlabel=cv_labels[0], ylabel=cv_labels[1],
                title=f"FPS clustering, {n_clusters} clusters, seed {seed}",
                filename=str(output_dir / "clustering.png"),
            )
            print(f"Saved to {output_dir}", flush=True)

    # ------------------------------------------------------------------
    # Mode: resample — one shared clustering, alternative representatives
    # ------------------------------------------------------------------
    else:
        print(f"\n{'=' * 70}", flush=True)
        print(f"Shared clustering (seed {seeds[0]})", flush=True)
        print(f"{'=' * 70}", flush=True)

        centers_idx, labels = fps_clustering(
            X, n_clusters,
            seed=seeds[0],
            backend=args.backend,
            refine=args.refine,
            n_iterations=args.n_iterations,
            pool_frac=args.pool_frac,
            verbose=True,
        )
        counts = np.bincount(labels, minlength=n_clusters)
        print(f"Cluster occupancy: min {counts.min()}, median "
              f"{int(np.median(counts))}, max {counts.max()}", flush=True)

        # Shared clustering at the root: read with gmm_build.py --shared_clusters
        save_cluster_assignment(cluster_root, centers_idx, labels)

        sample_idx = resample_cluster_members(
            labels, n_samples=args.n_sets, seed=args.resample_seed,
        )
        for i in range(args.n_sets):
            output_dir = cluster_root / f"set_{i}"
            os.makedirs(output_dir, exist_ok=True)
            if not args.no_pdb:
                for k, frame_id in enumerate(sample_idx[:, i]):
                    save_samples_to_pdb(
                        xyz[frame_id], traj.top, str(output_dir / f"sample_{k}.pdb"),
                    )
            np.savetxt(output_dir / "sample_idx.txt", sample_idx[:, i], fmt="%d")
            plot_fps_clustering(
                X, centers_idx, labels,
                highlight_idx=sample_idx[:, i],
                xlabel=cv_labels[0], ylabel=cv_labels[1],
                title=f"FPS clustering, {n_clusters} clusters, resampling set {i}",
                filename=str(output_dir / "sampling.png"),
            )
            print(f"Saved set {i} to {output_dir}", flush=True)

    print(f"\nDone. Pass --cluster_root {cluster_root} to gmm_build.py.", flush=True)
