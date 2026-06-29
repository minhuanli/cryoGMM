"""
Step 2b: Total Variation Distance (TVD) Heatmap

Computes and visualizes the TVD between probability distributions across
experimental conditions. The TVD combines the weighted monomer CV histogram
and the discrete state probabilities (dimer1, dimer2, junk/noise).

Three CVs are available (select with --cv_index):
  0 — distance between residues 50 and 120  (BB2 topology)
  1 — distance between residues 6  and 81   (BB2 topology)
  2 — angle    between residues 123, 99, 49 (BB2 topology, vertex=99)

Reads from the gmm_cvs/ directory produced by step1 (gmm_cvs.py):
  {root_dir}/{system_path}/{n_clusters}_centers/gmm_cvs/
    cvs_{job_id}.npy
    cvs_md.npy
    weights_md.npy
    weights_means_{job_id}.npy

Output:
    {root_dir}/{system_path}/{n_clusters}_centers/
        tvd_heatmap_{system}_N{K}_cv{i}.png / .svg
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TVD heatmap for GMM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory for GMM results.")
    parser.add_argument("--system_path", type=str, required=True,
                        help="System path relative to root_dir.")
    parser.add_argument("--n_clusters", type=int, default=40,
                        help="Number of clusters for GMM (default: 40).")
    parser.add_argument(
        "--cv_index", type=int, default=0, choices=[0, 1, 2],
        help=(
            "Which CV column to use for binning: "
            "0=dist(resi50, resi120), "
            "1=dist(resi6, resi81), "
            "2=angle(resi123, resi99, resi49) "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--job_ids", type=str, default=None,
        help="Comma-separated job IDs in display order. "
             "Default: 'md,1609,1628,1631,1605,1633,1632,1709,1840'.",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated condition labels matching --job_ids. "
             "Default: 'Prior,0,0.5,1,3,5,10,Mutant,Control'.",
    )
    return parser.parse_args()


# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_JOB_IDS   = ["md", 1609, 1628, 1631, 1605, 1633, 1632, 1709, 1840]
DEFAULT_MG_LABELS = ['Prior', '0', '0.5', '1', '3', '5', '10', 'Mutant', 'Control']

CV_BIN_MIN = 0
CV_N_BINS  = 6
CV_BIN_MAX = {0: 18.0, 1: 10.0, 2: 3.1416}

MONOMER_INDEX = 0
DIMER1_INDEX  = 1
DIMER2_INDEX  = 2
JUNK_INDEX    = 3

FIGURE_WIDTH      = 8
FIGURE_HEIGHT     = 7
DPI               = 200
COLORMAP          = 'Blues_r'
VMIN              = 0
VMAX              = 0.5
COLORBAR_FRACTION = 0.046
COLORBAR_PAD      = 0.04
TICK_FONTSIZE     = 14


# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == "__main__":
    args        = parse_args()
    ROOT_DIR    = args.root_dir
    SYSTEM_PATH = args.system_path
    N_CLUSTERS  = args.n_clusters
    CV_INDEX    = args.cv_index

    JOB_IDS   = DEFAULT_JOB_IDS
    MG_LABELS = DEFAULT_MG_LABELS
    if args.job_ids is not None:
        raw     = args.job_ids.split(",")
        JOB_IDS = ["md" if x == "md" else int(x) for x in raw]
    if args.labels is not None:
        MG_LABELS = args.labels.split(",")

    cv_bin_max = CV_BIN_MAX[CV_INDEX]

    INPUT_DIR  = os.path.join(ROOT_DIR, SYSTEM_PATH, f"{N_CLUSTERS}_centers/gmm_cvs")
    OUTPUT_DIR = os.path.join(ROOT_DIR, SYSTEM_PATH, f"{N_CLUSTERS}_centers")
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        f"tvd_heatmap_{SYSTEM_PATH.split('/')[1]}_N{N_CLUSTERS}_cv{CV_INDEX}"
    )

    print(f"Root dir: {ROOT_DIR}")
    print(f"System path: {SYSTEM_PATH}")
    print(f"N clusters: {N_CLUSTERS}")
    print(f"CV index: {CV_INDEX}  (bin max={cv_bin_max})")
    print(f"Input directory: {INPUT_DIR}")

    # ===== LOAD CV DATA =====
    print("\nLoading CV data...", flush=True)
    all_cvs = []
    for job_id in JOB_IDS:
        cvs = np.load(f"{INPUT_DIR}/cvs_{job_id}.npy", allow_pickle=True)
        if job_id == "md":
            cv_data = np.asarray(cvs)[:, CV_INDEX]
        else:
            cv_data = np.concatenate([np.asarray(cv)[:, CV_INDEX] for cv in cvs])
        all_cvs.append(cv_data)

    # ===== COMPUTE CV HISTOGRAMS =====
    print("Computing CV histograms...", flush=True)
    bins    = np.linspace(CV_BIN_MIN, cv_bin_max, CV_N_BINS)
    all_data = []
    for cv_data in all_cvs:
        hist, _ = np.histogram(cv_data, bins=bins, density=True)
        all_data.append(hist / hist.sum())
    all_data = np.array(all_data)

    # ===== LOAD WEIGHT STATISTICS =====
    print("Loading weight statistics...", flush=True)
    all_weights_means = []
    for job_id in JOB_IDS:
        if job_id == "md":
            weights_md = np.load(f"{INPUT_DIR}/weights_md.npy", allow_pickle=True)
            all_weights_means.append(weights_md)
        else:
            weights_mean = np.load(f"{INPUT_DIR}/weights_means_{job_id}.npy", allow_pickle=True)
            all_weights_means.append(weights_mean[0])
    all_weights_means = np.array(all_weights_means)

    # ===== COMBINE HISTOGRAMS WITH WEIGHTS =====
    print("Combining CV histograms with discrete state probabilities...", flush=True)
    all_hists = []
    for monomer_hist, all_weights in zip(all_data, all_weights_means):
        combined_hist = np.concatenate([
            monomer_hist * all_weights[MONOMER_INDEX],
            all_weights[DIMER1_INDEX:JUNK_INDEX + 1]
        ])
        all_hists.append(combined_hist)
    all_hists = np.array(all_hists)

    # ===== COMPUTE TVD MATRIX =====
    print("Computing TVD matrix...", flush=True)
    n_conditions = len(all_hists)
    tvd_matrix   = np.zeros((n_conditions, n_conditions))
    for i in range(n_conditions):
        for j in range(n_conditions):
            tvd_matrix[i, j] = 0.5 * np.sum(np.abs(all_hists[i] - all_hists[j]))
    print(f"Max TVD: {tvd_matrix.max():.4f}", flush=True)

    # ===== CREATE HEATMAP =====
    print("Creating TVD heatmap...", flush=True)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)

    cmap = plt.get_cmap(COLORMAP).copy()
    im   = ax.imshow(tvd_matrix, cmap=cmap, vmin=VMIN, vmax=VMAX)

    cbar = plt.colorbar(im, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
    cbar.set_label('Total Variation Distance', fontsize=TICK_FONTSIZE)

    ax.set_xticks(np.arange(len(MG_LABELS)))
    ax.set_yticks(np.arange(len(MG_LABELS)))
    ax.set_xticklabels(MG_LABELS, fontsize=TICK_FONTSIZE, rotation=60, ha='right')
    ax.set_yticklabels(MG_LABELS, fontsize=TICK_FONTSIZE)

    ax_xlabel = ax.set_xlabel("Condition", fontsize=TICK_FONTSIZE, fontweight='bold')
    ax_ylabel = ax.set_ylabel("Condition", fontsize=TICK_FONTSIZE, fontweight='bold')
    ax_title  = ax.set_title("Total Variation Distance Heatmap",
                              fontsize=TICK_FONTSIZE + 2, fontweight='bold')

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FILE}.png", dpi=DPI, bbox_inches='tight', transparent=True)
    print(f"\nSaved {OUTPUT_FILE}.png")

    cbar.ax.yaxis.label.set_visible(False)
    ax_xlabel.set_visible(False)
    ax_ylabel.set_visible(False)
    ax_title.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    cbar.set_ticks([])

    plt.savefig(f"{OUTPUT_FILE}.svg", dpi=DPI, bbox_inches='tight', transparent=True)
    print(f"Saved {OUTPUT_FILE}.svg")
    plt.close()

    print("\n=== TVD heatmap generated successfully! ===", flush=True)
