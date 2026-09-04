"""
Step 0: Build GMM — per-cluster local PCA + Ledoit-Wolf + bond filtering

Algorithm:
  - Per-cluster local PCA whitener with adaptive keepdims (95% variance threshold)
  - Ledoit-Wolf covariance estimation in the reduced PCA space
    Degeneracy threshold: N >= 2*keepdims (appropriate for Ledoit-Wolf)
  - Bond-distance filtering during sampling to remove geometrically implausible structures

All run-specific configuration is passed as CLI arguments.

Output structure:
  {output_root}/
    bond_dist_gaussian/
      mu_bond.npy
      sigma_bond.npy
    {n_clusters}_centers/
      Set_{set_id}/
        whiteners/
          whitener_cluster_{k}.pkl
        gmm_cluster_{k}_Mu_normed.pt
        gmm_cluster_{k}_Sigma.pt
        gmm_cluster_{k}_is_degenerate.npy
      Job_{job_id}/
        Set_{set_id}/
          gmm_pi.pt
          gmm_samples/
            gmm_sample_BB2_{i:03d}.pdb
            gmm_centers.txt
"""

import os
import pickle
import argparse
import torch
import mdtraj as md
import numpy as np

from cryogmm import Whitener
from cryogmm.utils import assert_tensor, assert_numpy
from cryogmm.models.gmm import sample_gmm_with_bond_filter, build_cluster_whitener_and_params
from cryogmm.utils import save_samples_to_pdb

from pathlib import Path
from tqdm import tqdm


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic GMM build: per-cluster local PCA + Ledoit-Wolf + bond filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- paths ---
    parser.add_argument(
        "--traj_path", type=str, required=True,
        help="Path to trajectory positions (.pt file, shape (N_frames, N_atoms, 3), nm units).",
    )
    parser.add_argument(
        "--traj_top", type=str, required=True,
        help="Path to topology PDB file.",
    )
    parser.add_argument(
        "--cluster_root", type=str, required=True,
        help="Path to cluster root directory. May contain {n_clusters} as a template.",
    )
    parser.add_argument(
        "--weights_path_template", type=str, required=True,
        help="Template path to SBI weight files. Must contain {set_id} and {job_id}. "
             "May also contain {n_clusters}.",
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Full resolved path for all outputs (no template).",
    )
    parser.add_argument(
        "--bb_selection", type=str, required=True,
        help="MDTraj selection string for backbone atoms (e.g. 'name BB2', 'name SUGR').",
    )
    parser.add_argument(
        "--alignment_selection", type=str, required=True,
        help="MDTraj selection string for trajectory superposition.",
    )

    # --- tunable parameters ---
    parser.add_argument("--n_clusters",             type=int,   default=40)
    parser.add_argument("--n_samples",              type=int,   default=200,
                        help="GMM samples to generate per job/set.")
    parser.add_argument("--cov_reg_min",            type=float, default=1e-4,
                        help="Minimum eigenvalue floor for Ledoit-Wolf covariance. "
                             "Use 1e-2 for trajectories stored in Angstroms.")
    parser.add_argument("--bond_sigma_cutoff",      type=float, default=4.0)
    parser.add_argument("--bond_oversample_factor", type=int,   default=20)

    # --- job control ---
    parser.add_argument("--job_ids", type=str, required=True,
                        help="Comma-separated job IDs.")
    parser.add_argument("--set_ids", type=str, default="0,1,2,3,4",
                        help="Comma-separated set IDs (default: 0,1,2,3,4).")
    parser.add_argument("--device",  type=str, default="cuda:0")
    parser.add_argument("--force",   action="store_true",
                        help="Recompute and overwrite cached whiteners/covariances.")
    parser.add_argument("--shared_clusters", action="store_true",
                        help="Load center_idx.npy / cluster_labels.npy from cluster_root "
                             "directly (not from cluster_root/set_{set_id}/). Use for "
                             "resampling runs where cluster assignments are shared across sets.")

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
def main():
    args = parse_args()

    job_ids    = np.array([int(x) for x in args.job_ids.split(",")])
    set_ids    = np.array([int(x) for x in args.set_ids.split(",")])
    device     = args.device
    force      = args.force
    n_clusters = args.n_clusters
    output_root = args.output_root

    print(f"Run config:", flush=True)
    print(f"  job_ids    : {job_ids}", flush=True)
    print(f"  set_ids    : {set_ids}", flush=True)
    print(f"  n_clusters : {n_clusters}", flush=True)
    print(f"  output_root: {output_root}", flush=True)
    print(f"  device     : {device}", flush=True)
    print(f"  cov_reg_min: {args.cov_reg_min}", flush=True)

    # ------------------------------------------------------------------
    # Load and align trajectory (once)
    # ------------------------------------------------------------------
    print("\nLoading and processing trajectory...", flush=True)
    traj_xyz = torch.load(args.traj_path, weights_only=True).numpy()
    pdb = md.load_pdb(args.traj_top)
    traj_obj = md.Trajectory(traj_xyz, pdb.top)
    traj_aligned = traj_obj.superpose(
        traj_obj, 0,
        atom_indices=traj_obj.top.select(args.alignment_selection),
    )
    traj_aligned_BB = traj_aligned.atom_slice(
        traj_aligned.top.select(args.bb_selection)
    )
    xyz     = traj_aligned_BB.xyz   # (N_total, N_atoms, 3)
    n_atoms = traj_aligned_BB.n_atoms

    # ------------------------------------------------------------------
    # Bond distance Gaussian (cached across all sets/jobs)
    # ------------------------------------------------------------------
    print("\nComputing bond distance Gaussian from full trajectory...", flush=True)
    os.makedirs(output_root, exist_ok=True)

    bond_dist_folder = Path(output_root) / "bond_dist_gaussian"
    os.makedirs(bond_dist_folder, exist_ok=True)

    mu_bond_path    = bond_dist_folder / "mu_bond.npy"
    sigma_bond_path = bond_dist_folder / "sigma_bond.npy"

    if mu_bond_path.exists() and sigma_bond_path.exists():
        mu_bond    = np.load(mu_bond_path,    allow_pickle=True)
        sigma_bond = np.load(sigma_bond_path, allow_pickle=True)
        print(f"Bond distance Gaussian loaded from cache: {bond_dist_folder}", flush=True)
    else:
        d_ip1_i = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=2)
        mu_bond    = d_ip1_i.mean(0)
        sigma_bond = d_ip1_i.std(0)
        np.save(mu_bond_path,    mu_bond)
        np.save(sigma_bond_path, sigma_bond)
        print(f"Bond distance Gaussian saved to: {bond_dist_folder}", flush=True)

    # Resolve {n_clusters} in cluster_root if present
    cluster_root     = Path(args.cluster_root.format(n_clusters=n_clusters))
    weights_template = args.weights_path_template

    # ------------------------------------------------------------------
    # Outer loop: set_id  (clustering changes per set → whiteners change)
    # ------------------------------------------------------------------
    for set_id in set_ids:
        print(f"\n{'='*70}", flush=True)
        print(f"Processing Set ID: {set_id}", flush=True)
        print(f"{'='*70}", flush=True)

        if args.shared_clusters:
            cluster_root_set = cluster_root
        else:
            cluster_root_set = cluster_root / f"set_{set_id}"
        center_id  = np.load(cluster_root_set / "center_idx.npy",    allow_pickle=True)
        cluster_id = np.load(cluster_root_set / "cluster_labels.npy", allow_pickle=True)

        set_shared_folder = Path(output_root) / f"{n_clusters}_centers" / f"Set_{set_id}"
        whitener_folder   = set_shared_folder / "whiteners"
        os.makedirs(whitener_folder, exist_ok=True)

        # ------------------------------------------------------------------
        # Per-cluster whitener + covariance (cached on disk)
        # ------------------------------------------------------------------
        print(f"\nBuilding per-cluster whiteners for Set {set_id}...", flush=True)
        whiteners          = []
        means_list         = []
        sigmas_list        = []
        is_degenerate_list = []

        for k in range(n_clusters):
            whitener_path        = whitener_folder  / f"whitener_cluster_{k}.pkl"
            mu_path              = set_shared_folder / f"gmm_cluster_{k}_Mu_normed.pt"
            sigma_path           = set_shared_folder / f"gmm_cluster_{k}_Sigma.pt"
            degenerate_flag_path = set_shared_folder / f"gmm_cluster_{k}_is_degenerate.npy"

            if not force and whitener_path.exists() and mu_path.exists() and sigma_path.exists():
                with open(whitener_path, "rb") as f:
                    D_whitener = pickle.load(f)
                wtner_k = Whitener.from_dict(D_whitener, device=device)
                mu_k    = torch.load(mu_path,    weights_only=True).to(device=device, dtype=torch.float32)
                Sigma_k = torch.load(sigma_path, weights_only=True).to(device=device, dtype=torch.float32)
                is_deg  = (
                    bool(np.load(degenerate_flag_path, allow_pickle=True))
                    if degenerate_flag_path.exists() else False
                )
                print(
                    f"  Cluster {k}: loaded from cache "
                    f"(dim={mu_k.shape[0]}, degenerate={is_deg})",
                    flush=True,
                )
            else:
                wtner_k, mu_k, Sigma_k, is_deg = build_cluster_whitener_and_params(
                    k, xyz, cluster_id, center_id, args.cov_reg_min, device
                )
                with open(whitener_path, "wb") as f:
                    pickle.dump(wtner_k.to_dict(), f)
                torch.save(mu_k,    mu_path)
                torch.save(Sigma_k, sigma_path)
                np.save(degenerate_flag_path, np.array(is_deg))

            whiteners.append(wtner_k)
            means_list.append(mu_k)
            sigmas_list.append(Sigma_k)
            is_degenerate_list.append(is_deg)

        print(f"All {n_clusters} cluster whiteners ready for Set {set_id}.", flush=True)

        # ------------------------------------------------------------------
        # Inner loop: job_id  (only SBI weights change per job)
        # ------------------------------------------------------------------
        for job_id in tqdm(job_ids, desc=f"Set {set_id} — Jobs"):
            print(f"\n--- Processing Job {job_id} / Set {set_id} ---", flush=True)

            weights_path = weights_template.format(
                job_id=job_id, set_id=set_id, n_clusters=n_clusters
            )
            if weights_path.endswith(".txt"):
                weights = torch.from_numpy(np.loadtxt(weights_path)).float()
            else:
                weights = torch.load(weights_path, weights_only=True)
            pi_raw  = weights[:n_clusters]
            pi      = (pi_raw / pi_raw.sum()).to(device=device, dtype=torch.float32)
            assert torch.isclose(pi.sum(), torch.tensor(1.0)), "Weights do not sum to 1"

            output_folder = (
                Path(output_root)
                / f"{n_clusters}_centers"
                / f"Job_{job_id}"
                / f"Set_{set_id}"
            )
            os.makedirs(output_folder, exist_ok=True)

            torch.save(pi, output_folder / "gmm_pi.pt")
            print(f"Weights saved to: {output_folder / 'gmm_pi.pt'}", flush=True)

            print(
                f"Generating {args.n_samples} samples "
                f"(oversample_factor={args.bond_oversample_factor})...",
                flush=True,
            )
            gmm_samples_xyz, gmm_ids = sample_gmm_with_bond_filter(
                pi=pi,
                means=means_list,
                covariances=sigmas_list,
                whiteners=whiteners,
                is_degenerate=is_degenerate_list,
                mu_bond=mu_bond,
                sigma_bond=sigma_bond,
                bond_sigma_cutoff=args.bond_sigma_cutoff,
                n_atoms=n_atoms,
                num_samples=args.n_samples,
                oversample_factor=args.bond_oversample_factor,
                device=device,
                cov_reg_min=args.cov_reg_min,
            )

            sample_folder = output_folder / "gmm_samples"
            os.makedirs(sample_folder, exist_ok=True)
            np.savetxt(sample_folder / "gmm_centers.txt", gmm_ids.cpu().numpy(), fmt="%d")

            for i in range(gmm_samples_xyz.shape[0]):
                save_samples_to_pdb(
                    gmm_samples_xyz[i],
                    traj_aligned_BB.top,
                    str(sample_folder / f"gmm_sample_BB2_{i:03d}.pdb"),
                )

            print(
                f"Job {job_id} / Set {set_id} complete. "
                f"Results saved to: {output_folder}",
                flush=True,
            )

    print("\n=== All sets and jobs processed successfully! ===", flush=True)


if __name__ == "__main__":
    main()
