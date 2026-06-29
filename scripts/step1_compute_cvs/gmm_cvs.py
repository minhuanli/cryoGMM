"""
Step 1: Compute GMM CVs and Weight Statistics

Computes three collective variables from GMM samples:
  CV1 — distance between residues 50 and 120
  CV2 — distance between residues 6  and 81
  CV3 — angle    between residues 123, 99, 49  (vertex=99)

CVs are stacked into shape (N_samples, 3) per set and saved per job.

Weight structure assumed: 4-class (monomer + dimer1 + dimer2 + noise).
The SBI weight vector has shape (N_clusters + 4,):
  weights[:N_clusters]   — per-cluster monomer mixing weights
  weights[N_clusters]    — dimer-1 weight
  weights[N_clusters+1]  — dimer-2 weight
  weights[N_clusters+2]  — noise/junk weight
  weights[N_clusters+3]  — (unused; included for compatibility with some decoders)

Output layout under --gmm_output_root:
    {gmm_output_root}/{n_clusters}_centers/gmm_cvs/
        cvs_{job_id}.npy            — object array of (N_samples, 3) per set
        weights_means_{job_id}.npy  — shape (1, 4): mean of [monomer, dimer1, dimer2, noise]
        weights_stds_{job_id}.npy   — shape (1, 4): std of the above
        cvs_md.npy                  — (N_SAMPLES, 3) MD reference CVs
        weights_md.npy              — (4,) uniform prior

Notes:
  • --cluster_root and --weights_path_template may contain {n_clusters},
    which is resolved to the value of --n_clusters at runtime.
  • Use --angstrom_to_nm for trajectories stored in Angstroms (divide coords by 10).
"""

import os
import pickle
import argparse
from pathlib import Path

import torch
import mdtraj as md
import numpy as np
from tqdm import tqdm

from cryogmm import Whitener
from cryogmm.models.gmm import sample_gmm_with_bond_filter
from cryogmm.preprocess import dist, angle


# ============================================================================
# CV RESIDUE INDICES (BB2 / SUGR / C1' backbone topology)
# ============================================================================
RESI_DIST1_A = 50   # CV1: dist(50, 120)
RESI_DIST1_B = 120
RESI_DIST2_A = 6    # CV2: dist(6, 81)
RESI_DIST2_B = 81
RESI_ANGLE_1 = 123  # CV3: angle(123, 99, 49)  vertex=99
RESI_ANGLE_2 = 99
RESI_ANGLE_3 = 49


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic GMM CV generator: samples GMM, computes CVs, saves statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- paths ---
    parser.add_argument("--traj_path", type=str, required=True,
                        help="Path to trajectory positions (.pt file).")
    parser.add_argument("--traj_top", type=str, required=True,
                        help="Path to topology PDB file.")
    parser.add_argument("--gmm_output_root", type=str, required=True,
                        help="Root directory where the built GMM lives (step-0 output).")
    parser.add_argument("--cluster_root", type=str, required=True,
                        help="Path to cluster centers directory. May contain {n_clusters}.")
    parser.add_argument("--weights_path_template", type=str, required=True,
                        help="Template for SBI weight files. Must contain {set_id} and {job_id}. "
                             "May also contain {n_clusters}.")
    # --- atom selections ---
    parser.add_argument("--bb_selection", type=str, required=True,
                        help="MDTraj selection for backbone atoms (e.g. 'name BB2').")
    parser.add_argument("--alignment_selection", type=str, required=True,
                        help="MDTraj selection for superposition.")
    # --- job control ---
    parser.add_argument("--job_ids", type=str, required=True,
                        help="Comma-separated job IDs.")
    parser.add_argument("--n_clusters", type=int, default=40,
                        help="Number of clusters (default: 40).")
    parser.add_argument("--set_ids", type=str, default="0,1,2,3,4",
                        help="Comma-separated set IDs (default: 0,1,2,3,4).")
    # --- sampling parameters ---
    parser.add_argument("--n_samples", type=int, default=20000,
                        help="MD reference samples and max GMM samples per job (default: 20000).")
    parser.add_argument("--cov_reg_min", type=float, default=1e-4,
                        help="Minimum eigenvalue floor for covariance regularisation. "
                             "Use 1e-2 for Angstrom-scale coordinates. Default: 1e-4.")
    parser.add_argument("--bond_sigma_cutoff", type=float, default=4.0)
    parser.add_argument("--bond_oversample_factor", type=int, default=20)
    # --- misc ---
    parser.add_argument("--angstrom_to_nm", action="store_true",
                        help="Divide trajectory coordinates by 10 (Angstrom → nm). "
                             "Use for trajectories not stored in nm.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--only_md", action="store_true",
                        help="Only regenerate MD CVs; skip GMM sampling.")
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    args = parse_args()

    DEVICE     = args.device
    N_CLUSTERS = args.n_clusters
    N_SAMPLES  = args.n_samples
    JOB_IDS    = [int(j) if j.isdigit() else j for j in args.job_ids.split(",")]
    SET_IDS    = [int(s) for s in args.set_ids.split(",")]

    COV_REG_MIN            = args.cov_reg_min
    BOND_SIGMA_CUTOFF      = args.bond_sigma_cutoff
    BOND_OVERSAMPLE_FACTOR = args.bond_oversample_factor

    # Resolve {n_clusters} templates
    CLUSTER_ROOT     = args.cluster_root.format(n_clusters=N_CLUSTERS)
    WEIGHTS_TEMPLATE = args.weights_path_template  # still has {set_id}, {job_id}

    GMM_OUTPUT_ROOT = args.gmm_output_root
    OUTPUT_DIR = os.path.join(GMM_OUTPUT_ROOT, f"{N_CLUSTERS}_centers", "gmm_cvs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"GMM output root : {GMM_OUTPUT_ROOT}", flush=True)
    print(f"n_clusters      : {N_CLUSTERS}", flush=True)
    print(f"job_ids         : {JOB_IDS}", flush=True)
    print(f"set_ids         : {SET_IDS}", flush=True)
    print(f"device          : {DEVICE}", flush=True)
    print(f"only_md         : {args.only_md}", flush=True)
    print(f"angstrom_to_nm  : {args.angstrom_to_nm}", flush=True)
    print(f"cov_reg_min     : {COV_REG_MIN}", flush=True)

    # ------------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------------
    print("\nLoading trajectory and topology...", flush=True)
    traj_xyz = torch.load(args.traj_path, weights_only=True).numpy()
    if args.angstrom_to_nm:
        traj_xyz = traj_xyz / 10.0
    pdb      = md.load_pdb(args.traj_top)
    traj_obj = md.Trajectory(traj_xyz, pdb.top)

    top_BB  = pdb.atom_slice(pdb.top.select(args.bb_selection))
    traj_BB = traj_obj.atom_slice(traj_obj.top.select(args.bb_selection))

    ind_d1a = int(top_BB.top.select(f"resi {RESI_DIST1_A}")[0])
    ind_d1b = int(top_BB.top.select(f"resi {RESI_DIST1_B}")[0])
    ind_d2a = int(top_BB.top.select(f"resi {RESI_DIST2_A}")[0])
    ind_d2b = int(top_BB.top.select(f"resi {RESI_DIST2_B}")[0])
    ind_a1  = int(top_BB.top.select(f"resi {RESI_ANGLE_1}")[0])
    ind_a2  = int(top_BB.top.select(f"resi {RESI_ANGLE_2}")[0])
    ind_a3  = int(top_BB.top.select(f"resi {RESI_ANGLE_3}")[0])

    # ------------------------------------------------------------------
    # MD reference CVs
    # ------------------------------------------------------------------
    md_ind   = np.random.choice(traj_BB.xyz.shape[0], size=N_SAMPLES)
    xyz_md   = traj_BB.xyz[md_ind]

    cv_md_d1  = dist( xyz_md[:, ind_d1a], xyz_md[:, ind_d1b])
    cv_md_d2  = dist( xyz_md[:, ind_d2a], xyz_md[:, ind_d2b])
    cv_md_ang = angle(xyz_md[:, ind_a1],  xyz_md[:, ind_a2], xyz_md[:, ind_a3])
    cv_md = np.stack([cv_md_d1, cv_md_d2, cv_md_ang], axis=1)

    weights_md = np.array([
        N_CLUSTERS / (N_CLUSTERS + 3.),
        1. / (N_CLUSTERS + 3.),
        1. / (N_CLUSTERS + 3.),
        1. / (N_CLUSTERS + 3.),
    ])

    np.save(os.path.join(OUTPUT_DIR, "cvs_md.npy"),     cv_md)
    np.save(os.path.join(OUTPUT_DIR, "weights_md.npy"), weights_md)
    print(f"Saved MD CVs → {OUTPUT_DIR}/cvs_md.npy", flush=True)

    if args.only_md:
        print("\n=== Only MD mode: skipping GMM CV generation ===", flush=True)
        exit(0)

    # ------------------------------------------------------------------
    # Load bond distance Gaussian (shared across all sets/jobs)
    # ------------------------------------------------------------------
    bond_dist_folder = Path(GMM_OUTPUT_ROOT) / "bond_dist_gaussian"
    mu_bond    = np.load(bond_dist_folder / "mu_bond.npy",    allow_pickle=True)
    sigma_bond = np.load(bond_dist_folder / "sigma_bond.npy", allow_pickle=True)
    print(f"\nBond distance Gaussian loaded from: {bond_dist_folder}", flush=True)

    traj_aligned = traj_obj.superpose(
        traj_obj, 0, atom_indices=traj_obj.top.select(args.alignment_selection)
    )
    traj_aligned_BB = traj_aligned.atom_slice(traj_aligned.top.select(args.bb_selection))
    n_atoms = traj_aligned_BB.n_atoms

    cluster_root = Path(CLUSTER_ROOT)
    all_cvs = {jid: [] for jid in JOB_IDS}

    # ------------------------------------------------------------------
    # Outer loop: set_id  (whiteners change with clustering)
    # ------------------------------------------------------------------
    for set_id in SET_IDS:
        print(f"\n{'='*70}", flush=True)
        print(f"Processing Set ID: {set_id}", flush=True)
        print(f"{'='*70}", flush=True)

        set_shared_folder = Path(GMM_OUTPUT_ROOT) / f"{N_CLUSTERS}_centers" / f"Set_{set_id}"
        whitener_folder   = set_shared_folder / "whiteners"

        print(f"Loading per-cluster whiteners for Set {set_id}...", flush=True)
        whiteners          = []
        means_list         = []
        sigmas_list        = []
        is_degenerate_list = []

        for k in range(N_CLUSTERS):
            whitener_path        = whitener_folder / f"whitener_cluster_{k}.pkl"
            mu_path              = set_shared_folder / f"gmm_cluster_{k}_Mu_normed.pt"
            sigma_path           = set_shared_folder / f"gmm_cluster_{k}_Sigma.pt"
            degenerate_flag_path = set_shared_folder / f"gmm_cluster_{k}_is_degenerate.npy"

            with open(whitener_path, "rb") as f:
                D_whitener = pickle.load(f)
            wtner_k  = Whitener.from_dict(D_whitener, device=DEVICE)
            mu_k     = torch.load(mu_path,    weights_only=True).to(device=DEVICE, dtype=torch.float32)
            Sigma_k  = torch.load(sigma_path, weights_only=True).to(device=DEVICE, dtype=torch.float32)
            is_deg   = bool(np.load(degenerate_flag_path, allow_pickle=True)) if degenerate_flag_path.exists() else False

            whiteners.append(wtner_k)
            means_list.append(mu_k)
            sigmas_list.append(Sigma_k)
            is_degenerate_list.append(is_deg)

        print(f"  Loaded {N_CLUSTERS} cluster whiteners.", flush=True)

        # ------------------------------------------------------------------
        # Inner loop: job_id
        # ------------------------------------------------------------------
        for job_id in tqdm(JOB_IDS, desc=f"Set {set_id} — Jobs"):
            weights_path = WEIGHTS_TEMPLATE.format(
                job_id=job_id, set_id=set_id, n_clusters=N_CLUSTERS
            )
            weights = torch.load(weights_path, weights_only=True).numpy()

            pi_raw      = torch.tensor(weights[:N_CLUSTERS], dtype=torch.float32)
            monomer_sum = float(pi_raw.sum())
            pi          = (pi_raw / pi_raw.sum()).to(device=DEVICE)

            n_samples_job = max(int(N_SAMPLES * monomer_sum), 1)

            gmm_samples_xyz, _ = sample_gmm_with_bond_filter(
                pi=pi,
                means=means_list,
                covariances=sigmas_list,
                whiteners=whiteners,
                is_degenerate=is_degenerate_list,
                mu_bond=mu_bond,
                sigma_bond=sigma_bond,
                bond_sigma_cutoff=BOND_SIGMA_CUTOFF,
                n_atoms=n_atoms,
                num_samples=n_samples_job,
                oversample_factor=BOND_OVERSAMPLE_FACTOR,
                device=DEVICE,
                cov_reg_min=COV_REG_MIN,
            )

            # If trajectory was stored in Angstroms, GMM samples are in Angstroms too.
            # Convert to nm to match MD reference CVs.
            if args.angstrom_to_nm:
                gmm_samples_xyz = gmm_samples_xyz / 10.0

            cv_d1  = dist( gmm_samples_xyz[:, ind_d1a], gmm_samples_xyz[:, ind_d1b])
            cv_d2  = dist( gmm_samples_xyz[:, ind_d2a], gmm_samples_xyz[:, ind_d2b])
            cv_ang = angle(gmm_samples_xyz[:, ind_a1],  gmm_samples_xyz[:, ind_a2],
                           gmm_samples_xyz[:, ind_a3])
            cv_stack = np.stack([cv_d1, cv_d2, cv_ang], axis=1)
            all_cvs[job_id].append(cv_stack)

    # ------------------------------------------------------------------
    # Weight statistics and save
    # ------------------------------------------------------------------
    print("\nComputing weight statistics and saving results...", flush=True)
    for job_id in JOB_IDS:
        weights_list = []
        for set_id in SET_IDS:
            weights_path = WEIGHTS_TEMPLATE.format(
                job_id=job_id, set_id=set_id, n_clusters=N_CLUSTERS
            )
            tmp = torch.load(weights_path, weights_only=True).numpy()
            weights_list.append(tmp)

        weights_arr = np.stack(weights_list, axis=0)  # (n_sets, N_CLUSTERS+4)
        monomer_w   = weights_arr[:, :N_CLUSTERS].sum(axis=1)
        dimer1_w    = weights_arr[:, N_CLUSTERS]
        dimer2_w    = weights_arr[:, N_CLUSTERS + 1]
        noise_w     = weights_arr[:, N_CLUSTERS + 2]

        means = [(np.mean(monomer_w), np.mean(dimer1_w), np.mean(dimer2_w), np.mean(noise_w))]
        stds  = [(np.std(monomer_w),  np.std(dimer1_w),  np.std(dimer2_w),  np.std(noise_w))]

        np.save(os.path.join(OUTPUT_DIR, f"cvs_{job_id}.npy"),
                np.array(all_cvs[job_id], dtype=object))
        np.save(os.path.join(OUTPUT_DIR, f"weights_means_{job_id}.npy"), np.array(means))
        np.save(os.path.join(OUTPUT_DIR, f"weights_stds_{job_id}.npy"),  np.array(stds))

        print(f"  Job {job_id}: saved CVs ({len(all_cvs[job_id])} sets) → {OUTPUT_DIR}", flush=True)

    print("\n=== All jobs processed successfully! ===", flush=True)
