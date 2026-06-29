"""
Step 2c: CV Trend Plot

Creates two-panel trend plots showing:
- Top panel:    Average CV value vs a quantitative condition (e.g. Mg²⁺ concentration)
- Bottom panel: Dimer formation percentage vs the same condition

Three CVs are available (select with --cv_index):
  0 — distance between residues 50 and 120  (BB2 topology)
  1 — distance between residues 6  and 81   (BB2 topology)
  2 — angle    between residues 123, 99, 49 (BB2 topology, vertex=99)

Reads from the gmm_cvs/ directory produced by step1 (gmm_cvs.py):
  {root_dir}/{system_path}/{n_clusters}_centers/gmm_cvs/
    cvs_{job_id}.npy
    weights_means_{job_id}.npy
    weights_stds_{job_id}.npy

Output:
    {root_dir}/{system_path}/{n_clusters}_centers/
        cv_trends_{system}_N{K}_cv{i}.png / .svg
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
        description="Generate CV trend plots for GMM analysis",
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
            "Which CV column to plot: "
            "0=dist(resi50, resi120), "
            "1=dist(resi6, resi81), "
            "2=angle(resi123, resi99, resi49) "
            "(default: 0)"
        ),
    )
    parser.add_argument(
        "--job_ids", type=str, default=None,
        help="Comma-separated job IDs in display order. "
             "Default: '1609,1628,1631,1605,1633,1632'.",
    )
    parser.add_argument(
        "--x_values", type=str, default=None,
        help="Comma-separated x-axis values matching --job_ids (e.g. Mg²⁺ mM). "
             "Default: '0.0,0.5,1.0,3.0,5.0,10.0'.",
    )
    parser.add_argument(
        "--xlabel", type=str, default="Mg²⁺ Concentration (mM)",
        help="X-axis label (default: 'Mg²⁺ Concentration (mM)').",
    )
    return parser.parse_args()


# ============================================================================
# CONFIGURATION: DATA PARAMETERS
# ============================================================================
DEFAULT_JOB_IDS          = [1609, 1628, 1631, 1605, 1633, 1632]
DEFAULT_MG_CONCENTRATIONS = [0.0, 0.5, 1.0, 3.0, 5.0, 10.0]

COLORS = [
    '#c7e9c0',  # 0 mM    - light green
    '#74c476',  # 0.5 mM  - light green
    '#7fcdbb',  # 1 mM    - cyan
    '#41b6c4',  # 3 mM    - teal
    '#2c7fb8',  # 5 mM    - blue
    '#253494',  # 10 mM   - dark blue
]

DIMER1_INDEX = 1
DIMER2_INDEX = 2


# ============================================================================
# CONFIGURATION: PLOT PARAMETERS
# ============================================================================
FIGURE_WIDTH  = 2.2
FIGURE_HEIGHT = 3.5
DPI           = 150

MARKER_DISTANCE = 'o'
MARKER_DIMER    = 's'
MARKER_SIZE     = 4
LINE_WIDTH      = 1
CAPSIZE         = 3
CAPTHICK        = 1
PLOT_COLOR      = 'black'
GRID_ALPHA      = 0.3

YLABEL_DIMER = 'Dimer \nPercentage(%)'
YLIMIT_DIMER = (-5, 75)

CV_CONFIGS = {
    0: {
        "ylabel": "dist(resi50 – resi120) (nm)",
        "title":  "Average dist(resi50 – resi120)",
        "ylim":   (-1.0, 14.0),
    },
    1: {
        "ylabel": "dist(resi6 – resi81) (nm)",
        "title":  "Average dist(resi6 – resi81)",
        "ylim":   (0.0, 10.0),
    },
    2: {
        "ylabel": "angle(resi123–resi99–resi49) (rad)",
        "title":  "Average angle(resi123–resi99–resi49)",
        "ylim":   (0.0, 3.1416),
    },
}

FONTSIZE_XLABEL = 12
FONTSIZE_YLABEL = 12
FONTSIZE_TITLE  = 14


# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == "__main__":
    args = parse_args()
    ROOT_DIR    = args.root_dir
    SYSTEM_PATH = args.system_path
    N_CLUSTERS  = args.n_clusters
    CV_INDEX    = args.cv_index

    JOB_IDS           = DEFAULT_JOB_IDS
    MG_CONCENTRATIONS = DEFAULT_MG_CONCENTRATIONS
    if args.job_ids is not None:
        JOB_IDS = [int(j) for j in args.job_ids.split(",")]
    if args.x_values is not None:
        MG_CONCENTRATIONS = [float(v) for v in args.x_values.split(",")]

    cv_cfg          = CV_CONFIGS[CV_INDEX]
    YLABEL_DISTANCE = cv_cfg["ylabel"]
    TITLE_DISTANCE  = cv_cfg["title"]
    YLIMIT_DISTANCE = cv_cfg["ylim"]

    INPUT_DIR  = os.path.join(ROOT_DIR, SYSTEM_PATH, f"{N_CLUSTERS}_centers/gmm_cvs")
    OUTPUT_DIR = os.path.join(ROOT_DIR, SYSTEM_PATH, f"{N_CLUSTERS}_centers")
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR, f"cv_trends_{SYSTEM_PATH.split('/')[1]}_N{N_CLUSTERS}_cv{CV_INDEX}"
    )

    print(f"Root dir: {ROOT_DIR}")
    print(f"System path: {SYSTEM_PATH}")
    print(f"N clusters: {N_CLUSTERS}")
    print(f"CV index: {CV_INDEX}  ({YLABEL_DISTANCE})")
    print(f"Input directory: {INPUT_DIR}")

    # ===== LOAD CV DATA =====
    print("\nLoading CV data...", flush=True)
    all_cvs = []
    for job_id in JOB_IDS:
        cvs     = np.load(f"{INPUT_DIR}/cvs_{job_id}.npy", allow_pickle=True)
        cv_data = np.concatenate([np.asarray(cv)[:, CV_INDEX] for cv in cvs])
        all_cvs.append(cv_data)

    # ===== COMPUTE CV STATISTICS =====
    print("Computing CV statistics...", flush=True)
    all_data = np.array([[cv.mean(), cv.std()] for cv in all_cvs])

    # ===== COMPUTE DIMER STATISTICS =====
    print("Computing dimer statistics...", flush=True)
    dimer_means = []
    dimer_stds  = []
    for job_id in JOB_IDS:
        wm = np.load(f"{INPUT_DIR}/weights_means_{job_id}.npy", allow_pickle=True)[0]
        ws = np.load(f"{INPUT_DIR}/weights_stds_{job_id}.npy",  allow_pickle=True)[0]
        dimer_means.append(wm[DIMER1_INDEX] + wm[DIMER2_INDEX])
        dimer_stds.append(np.sqrt(ws[DIMER1_INDEX]**2 + ws[DIMER2_INDEX]**2))

    dimer_percentages     = [x * 100 for x in dimer_means]
    dimer_percentage_stds = [x * 100 for x in dimer_stds]

    # ===== CREATE PLOTS =====
    print("Creating plots...", flush=True)
    colors = COLORS[:len(JOB_IDS)] if len(JOB_IDS) <= len(COLORS) else COLORS * (len(JOB_IDS) // len(COLORS) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)

    # Top panel: CV value vs condition
    dist_lower_errs = [min(m, s) for m, s in zip(all_data[:, 0], all_data[:, 1])]
    ax1.plot(MG_CONCENTRATIONS, all_data[:, 0], color=PLOT_COLOR, linewidth=LINE_WIDTH, zorder=1)
    for x, y, elo, ehi, c in zip(MG_CONCENTRATIONS, all_data[:, 0], dist_lower_errs, all_data[:, 1], colors):
        ax1.errorbar(x, y, yerr=[[elo], [ehi]], fmt='none', color=PLOT_COLOR,
                     capsize=CAPSIZE, capthick=CAPTHICK, zorder=2)
        ax1.plot(x, y, marker=MARKER_DISTANCE, markersize=MARKER_SIZE + 2,
                 color=c, markeredgecolor=PLOT_COLOR, markeredgewidth=1.0, zorder=3)
    ax1_ylabel = ax1.set_ylabel(YLABEL_DISTANCE, fontsize=FONTSIZE_YLABEL)
    ax1_title  = ax1.set_title(TITLE_DISTANCE, fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_ylim(YLIMIT_DISTANCE)

    # Bottom panel: dimer percentage vs condition
    dimer_lower_errs = [min(m, s) for m, s in zip(dimer_percentages, dimer_percentage_stds)]
    ax2.plot(MG_CONCENTRATIONS, dimer_percentages, color=PLOT_COLOR, linewidth=LINE_WIDTH, zorder=1)
    for x, y, elo, ehi, c in zip(MG_CONCENTRATIONS, dimer_percentages, dimer_lower_errs, dimer_percentage_stds, colors):
        ax2.errorbar(x, y, yerr=[[elo], [ehi]], fmt='none', color=PLOT_COLOR,
                     capsize=CAPSIZE, capthick=CAPTHICK, zorder=2)
        ax2.plot(x, y, marker=MARKER_DIMER, markersize=MARKER_SIZE + 2,
                 color=c, markeredgecolor=PLOT_COLOR, markeredgewidth=1.0, zorder=3)
    ax2_xlabel = ax2.set_xlabel(args.xlabel, fontsize=FONTSIZE_XLABEL, fontweight='bold')
    ax2_ylabel = ax2.set_ylabel(YLABEL_DIMER, fontsize=FONTSIZE_YLABEL)
    ax2_title  = ax2.set_title("Dimer Formation", fontsize=FONTSIZE_TITLE, fontweight='bold')
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_ylim(YLIMIT_DIMER)

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FILE}.png", dpi=DPI, bbox_inches='tight', transparent=True)
    print(f"\nSaved {OUTPUT_FILE}.png")

    # Hide labels for clean SVG
    ax1_ylabel.set_visible(False)
    ax1_title.set_visible(False)
    ax2_xlabel.set_visible(False)
    ax2_ylabel.set_visible(False)
    ax2_title.set_visible(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.savefig(f"{OUTPUT_FILE}.svg", dpi=DPI, bbox_inches='tight', transparent=True)
    print(f"Saved {OUTPUT_FILE}.svg")
    plt.close()

    print("\n=== Plots generated successfully! ===", flush=True)
