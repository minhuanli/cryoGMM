"""
Step 2a: Stacked Probability Flow Plot

Creates stacked distribution plots showing:
- Left panel:  Bar plots of discrete state probabilities (Dimer1, Dimer2, Junk, Monomer)
- Right panel: Histogram of a chosen CV with ±std envelope across conditions

Three CVs are available (select with --cv_index):
  0 — distance between residues 50 and 120  (BB2 topology)
  1 — distance between residues 6  and 81   (BB2 topology)
  2 — angle    between residues 123, 99, 49 (BB2 topology, vertex=99)

Reads from the gmm_cvs/ directory produced by step1 (gmm_cvs.py):
  {root_dir}/{system_path}/{n_clusters}_centers/gmm_cvs/
    cvs_{job_id}.npy            — object array of (N_per_set, 3) arrays
    cvs_md.npy                  — (N, 3) MD reference
    weights_md.npy              — (4,) uniform prior weights
    weights_means_{job_id}.npy  — (1, 4) mean weights
    weights_stds_{job_id}.npy   — (1, 4) std weights

Output:
    {root_dir}/{system_path}/{n_clusters}_centers/
        gmm_dist_graph_flow_{system}_N{K}_cv{i}.png / .svg
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stacked probability flow plots for GMM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory for GMM results.",
    )
    parser.add_argument(
        "--system_path", type=str, required=True,
        help="System path relative to root_dir (e.g. 'martini/martini_fullcg_CV2_RMSD_FPS').",
    )
    parser.add_argument(
        "--n_clusters", type=int, default=40,
        help="Number of clusters for GMM (default: 40).",
    )
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
             "The special value 'md' loads cvs_md.npy and weights_md.npy. "
             "Default: 'md,1632,1633,1605,1631,1628,1609,1709,1840'.",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Comma-separated condition labels matching --job_ids. "
             "Default: 'Prior,10mM,5mM,3mM,1mM,0.5mM,0mM,Mutant,Control'.",
    )
    parser.add_argument(
        "--xlim_min", type=float, default=None,
        help="Override minimum x-axis limit (default: auto per CV type).",
    )
    parser.add_argument(
        "--xlim_max", type=float, default=None,
        help="Override maximum x-axis limit (default: auto per CV type).",
    )
    return parser.parse_args()


# ============================================================================
# CONFIGURATION: DATA PARAMETERS
# ============================================================================
DEFAULT_JOB_IDS = ["md", 1632, 1633, 1605, 1631, 1628, 1609, 1709, 1840]
DEFAULT_LABELS  = ['Prior', '10mM', '5mM', '3mM', '1mM', '0.5mM', '0mM', 'Mutant', 'Control']

COLORS = [
    '#000000',  # Prior   - black
    '#253494',  # 10 mM   - dark blue
    '#2c7fb8',  # 5 mM    - blue
    '#41b6c4',  # 3 mM    - teal
    '#7fcdbb',  # 1 mM    - cyan
    '#74c476',  # 0.5 mM  - light green
    '#c7e9c0',  # 0 mM    - lighter green
    '#969696',  # Mutant  - lighter gray
    '#636363',  # Control - gray
]


# ============================================================================
# CONFIGURATION: PLOT PARAMETERS
# ============================================================================
FIGURE_WIDTH  = 5
FIGURE_HEIGHT = 7
DPI           = 300
WIDTH_RATIOS  = [1, 2]
WSPACE        = 0.1

VERTICAL_SPACING  = 1.0
CONDITION_X_OFFSET = 1.5

HIST_BINS = 50

BAR_CATEGORIES         = ['Dimer1', 'Dimer2', 'Junk', 'Monomer']
BAR_WIDTH              = 0.4
BAR_ALPHA_NORMAL       = 0.8
BAR_ALPHA_DECOY_NOISE  = 0.3
DECOY_NOISE_INDICES    = [2]   # index of Junk in BAR_CATEGORIES

TICK_VALUES      = [0.0, 0.3, 0.6]
HIST_GRID_VALUES = [0.45]
TICK_FONTSIZE    = 6

REF_HEIGHT_FACTOR = 0.8

MEAN_LINE_STYLE = ':'
MEAN_LINE_WIDTH = 2.0
MEAN_LINE_ALPHA = 1.0
MEAN_LINE_COLOR = 'red'

GRID_LINE_STYLE = '--'
GRID_LINE_WIDTH = 0.8
GRID_LINE_ALPHA = 0.4
GRID_LINE_COLOR = 'grey'

YLABEL_BAR  = 'Probability'
YLABEL_HIST = 'Probability Density Function'

CV_CONFIGS = {
    0: {"xlabel": "dist(resi50 – resi120) (nm)",         "xlim": (0.0, 18.0)},
    1: {"xlabel": "dist(resi6 – resi81) (nm)",           "xlim": (0.0, 10.0)},
    2: {"xlabel": "angle(resi123–resi99–resi49) (rad)",  "xlim": (0.0, 3.1416)},
}


# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == "__main__":
    args = parse_args()
    ROOT_DIR    = args.root_dir
    SYSTEM_PATH = args.system_path
    N_CLUSTERS  = args.n_clusters
    CV_INDEX    = args.cv_index

    JOB_IDS = DEFAULT_JOB_IDS
    LABELS  = DEFAULT_LABELS
    if args.job_ids is not None:
        raw = args.job_ids.split(",")
        JOB_IDS = ["md" if x == "md" else int(x) for x in raw]
    if args.labels is not None:
        LABELS = args.labels.split(",")

    cv_cfg   = CV_CONFIGS[CV_INDEX]
    xlim_min = args.xlim_min if args.xlim_min is not None else cv_cfg["xlim"][0]
    xlim_max = args.xlim_max if args.xlim_max is not None else cv_cfg["xlim"][1]
    XLIM     = (xlim_min, xlim_max)
    XLABEL   = cv_cfg["xlabel"]

    INPUT_DIR  = os.path.join(ROOT_DIR, SYSTEM_PATH, f"{N_CLUSTERS}_centers/gmm_cvs")
    OUTPUT_DIR = os.path.join(ROOT_DIR, SYSTEM_PATH, f"{N_CLUSTERS}_centers")
    OUTPUT_FILE = os.path.join(
        OUTPUT_DIR,
        "gmm_dist_graph_flow_"
        + SYSTEM_PATH.split("/")[1]
        + f"_N{N_CLUSTERS}_cv{CV_INDEX}",
    )

    print(f"Root dir: {ROOT_DIR}")
    print(f"System path: {SYSTEM_PATH}")
    print(f"N clusters: {N_CLUSTERS}")
    print(f"CV index: {CV_INDEX}  ({XLABEL})")
    print(f"X-axis limits: {XLIM}")
    print(f"Input directory: {INPUT_DIR}")

    # ===== LOAD CV DATA =====
    print("\nLoading CV data...", flush=True)
    all_cvs = []
    for job_id in JOB_IDS:
        cvs = np.load(f"{INPUT_DIR}/cvs_{job_id}.npy", allow_pickle=True)
        if job_id == "md":
            per_set_cvs = [np.asarray(cvs)[:, CV_INDEX]]
        else:
            per_set_cvs = [np.asarray(cv)[:, CV_INDEX] for cv in cvs]
        all_cvs.append(per_set_cvs)

    # ===== LOAD WEIGHT DATA =====
    print("Loading weight data...", flush=True)
    all_weights_means = []
    all_weights_stds  = []
    for job_id in JOB_IDS:
        if job_id == "md":
            weights_md = np.load(f"{INPUT_DIR}/weights_md.npy", allow_pickle=True)
            all_weights_means.append(np.array([weights_md]))
            all_weights_stds.append(np.array([np.zeros_like(weights_md)]))
        else:
            all_weights_means.append(np.load(f"{INPUT_DIR}/weights_means_{job_id}.npy", allow_pickle=True))
            all_weights_stds.append(np.load(f"{INPUT_DIR}/weights_stds_{job_id}.npy",  allow_pickle=True))

    n_files = len(LABELS)

    # ===== CREATE STACKED PLOT =====
    def create_stacked_plot(output_file, bins=HIST_BINS, xlim=None):
        fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI, constrained_layout=True)
        gs  = gridspec.GridSpec(1, 2, width_ratios=WIDTH_RATIOS, wspace=WSPACE, figure=fig)
        ax_hist = plt.subplot(gs[1])
        ax_bar  = plt.subplot(gs[0])

        if xlim:
            global_min, global_max = xlim
        else:
            global_min = min([s.min() for per_set in all_cvs for s in per_set])
            global_max = max([s.max() for per_set in all_cvs for s in per_set])

        bin_edges  = np.linspace(global_min, global_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ref_height  = VERTICAL_SPACING * REF_HEIGHT_FACTOR

        condition_labels = []

        for i in range(n_files):
            cv_data       = all_cvs[i]
            weights       = all_weights_means[i][0]
            weights_std   = all_weights_stds[i][0]
            monomer_weight = weights[0]
            other_weights  = np.concatenate([weights[1:4], [weights[0]]])
            other_stds     = np.concatenate([weights_std[1:4], [weights_std[0]]])

            per_set_hists = np.array([
                np.histogram(cv_set, bins=bin_edges, density=False)[0]
                for cv_set in cv_data
            ])
            mean_hist = per_set_hists.mean(axis=0)
            std_hist  = per_set_hists.std(axis=0)

            norm_factor    = mean_hist.sum() if mean_hist.sum() > 0 else 1.0
            mean_hist_norm = (mean_hist / norm_factor) * ref_height * monomer_weight * 3.0
            std_hist_norm  = (std_hist  / norm_factor) * ref_height * monomer_weight * 3.0

            x_extended    = np.concatenate([[global_min], bin_centers, [global_max]])
            hist_extended = np.concatenate([[mean_hist_norm[0]], mean_hist_norm, [mean_hist_norm[-1]]])
            std_extended  = np.concatenate([[std_hist_norm[0]],  std_hist_norm,  [std_hist_norm[-1]]])

            mean_value = np.mean(np.concatenate(cv_data))

            j        = n_files - 1 - i
            y_offset = j * VERTICAL_SPACING

            ax_hist.plot(x_extended, hist_extended + y_offset, color='white', lw=2.8, zorder=1)
            ax_hist.fill_between(x_extended, hist_extended + y_offset, y_offset,
                                 color=COLORS[i % len(COLORS)], alpha=0.8, lw=0, zorder=2)

            upper_env = hist_extended + std_extended + y_offset
            lower_env = np.maximum(hist_extended - std_extended + y_offset, y_offset)
            ax_hist.fill_between(x_extended, upper_env, lower_env,
                                 color=COLORS[i % len(COLORS)], alpha=0.4, lw=0, zorder=2)
            ax_hist.plot(x_extended, upper_env, color=COLORS[i % len(COLORS)],
                         lw=0.8, linestyle='--', alpha=0.7, zorder=3)
            ax_hist.plot(x_extended, lower_env, color=COLORS[i % len(COLORS)],
                         lw=0.8, linestyle='--', alpha=0.7, zorder=3)
            ax_hist.plot(x_extended, hist_extended + y_offset, color=COLORS[i % len(COLORS)],
                         lw=1.2, zorder=3)

            ylim_min   = -0.3
            ylim_max   = n_files * VERTICAL_SPACING + 0.3
            line_start = y_offset
            line_end   = y_offset + 0.6 * ref_height
            ymin_norm  = (line_start - ylim_min) / (ylim_max - ylim_min)
            ymax_norm  = (line_end   - ylim_min) / (ylim_max - ylim_min)
            ax_hist.axvline(mean_value, ymin=ymin_norm, ymax=ymax_norm,
                            color=MEAN_LINE_COLOR, linestyle=MEAN_LINE_STYLE,
                            lw=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA, zorder=4)

            lbl = ax_bar.text(-2.8, y_offset + 0.35, LABELS[i],
                              va='center', ha='right', fontsize=9,
                              fontweight='bold', color='black')
            condition_labels.append(lbl)

            bar_positions = np.arange(len(BAR_CATEGORIES))
            for k, (bar_pos, weight, std) in enumerate(zip(bar_positions, other_weights, other_stds)):
                bar_h  = ref_height * weight
                alpha  = BAR_ALPHA_DECOY_NOISE if k in DECOY_NOISE_INDICES else BAR_ALPHA_NORMAL
                rect   = Rectangle((bar_pos - BAR_WIDTH/2, y_offset), BAR_WIDTH, bar_h,
                                    facecolor=COLORS[i % len(COLORS)], edgecolor='white',
                                    linewidth=1.5, alpha=alpha, zorder=2)
                ax_bar.add_patch(rect)
                ls  = ':' if k in DECOY_NOISE_INDICES else '-'
                rect_outline = Rectangle((bar_pos - BAR_WIDTH/2, y_offset), BAR_WIDTH, bar_h,
                                          facecolor='none', edgecolor=COLORS[i % len(COLORS)],
                                          linewidth=1.0, linestyle=ls, zorder=3)
                ax_bar.add_patch(rect_outline)
                if std > 0:
                    std_h = ref_height * std
                    ea    = BAR_ALPHA_DECOY_NOISE if k in DECOY_NOISE_INDICES else BAR_ALPHA_NORMAL
                    ax_bar.plot([bar_pos, bar_pos],
                                [y_offset + bar_h - std_h, y_offset + bar_h + std_h],
                                color='black', linewidth=1.2, zorder=5, alpha=ea)
                    cap_w = BAR_WIDTH * 0.4
                    for cap_y in [y_offset + bar_h + std_h, y_offset + bar_h - std_h]:
                        ax_bar.plot([bar_pos - cap_w/2, bar_pos + cap_w/2], [cap_y, cap_y],
                                    color='black', linewidth=1.2, zorder=5, alpha=ea)

        # ===== LEFT PANEL FINAL TOUCHES =====
        ax_hist.set_xlim(global_min, global_max)
        ax_hist.set_ylim(-0.3, n_files * VERTICAL_SPACING + 0.3)
        ax_hist.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))
        ax_hist.set_yticks([])
        ax_hist.set_yticklabels([])
        for spine in ax_hist.spines.values():
            spine.set_visible(False)
        ax_hist.tick_params(left=True, bottom=False, top=False, right=False)
        hist_ylabel = ax_hist.set_ylabel(YLABEL_HIST, fontsize=9, fontweight='bold')

        for j in range(n_files):
            for prob_value in HIST_GRID_VALUES:
                y_pos = j * VERTICAL_SPACING + prob_value * ref_height
                ax_hist.axhline(y=y_pos, color=GRID_LINE_COLOR, linestyle=GRID_LINE_STYLE,
                                linewidth=GRID_LINE_WIDTH, alpha=GRID_LINE_ALPHA, zorder=0.5)

        fig.canvas.draw()
        x_tick_values = [x for x in ax_hist.get_xticks() if global_min <= x <= global_max]
        tick_length   = 0.04
        for j in range(n_files):
            y_baseline = j * VERTICAL_SPACING
            for x_val in x_tick_values:
                if j == 0:
                    ax_hist.plot([x_val, x_val], [y_baseline, -0.15],
                                 color='black', linewidth=0.8, clip_on=False, zorder=10)
                else:
                    ax_hist.plot([x_val, x_val], [y_baseline, y_baseline - tick_length],
                                 color='black', linewidth=0.8, clip_on=False, zorder=10)

        ax_hist.tick_params(axis='x', which='both', bottom=False, top=False,
                            labelbottom=True, labelsize=TICK_FONTSIZE, pad=0.5)
        ax_hist.set_xticks(x_tick_values)
        ax_hist.spines['bottom'].set_visible(False)
        hist_xlabel = ax_hist.set_xlabel(XLABEL, fontsize=9, fontweight='bold')

        # ===== RIGHT PANEL FINAL TOUCHES =====
        ax_bar.set_xlim(-0.5, len(BAR_CATEGORIES) - 0.5)
        ax_bar.set_ylim(-0.3, n_files * VERTICAL_SPACING + 0.3)

        y_tick_positions_bar = []
        y_tick_labels_bar    = []
        for j in range(n_files):
            for prob_value in TICK_VALUES:
                y_pos = j * VERTICAL_SPACING + prob_value * ref_height
                y_tick_positions_bar.append(y_pos)
                y_tick_labels_bar.append(f'{prob_value:.1f}')

        ax_bar.set_yticks(y_tick_positions_bar)
        ax_bar.set_yticklabels(y_tick_labels_bar, fontsize=TICK_FONTSIZE)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.tick_params(left=True, bottom=False, top=False, right=False)
        bar_ylabel = ax_bar.set_ylabel(YLABEL_BAR, fontsize=9, fontweight='bold')

        for j in range(n_files):
            for prob_value in TICK_VALUES:
                y_pos = j * VERTICAL_SPACING + prob_value * ref_height
                ax_bar.axhline(y=y_pos, color=GRID_LINE_COLOR, linestyle=GRID_LINE_STYLE,
                               linewidth=GRID_LINE_WIDTH, alpha=GRID_LINE_ALPHA, zorder=1)

        ax_bar.set_xticks([])
        ax_bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        bar_category_labels = []
        for k, cat in enumerate(BAR_CATEGORIES):
            lbl = ax_bar.text(k - 0.2, -0.1, cat, ha='center', va='top',
                              fontsize=8, fontweight='bold', rotation=60)
            bar_category_labels.append(lbl)

        # Save PNG with all labels
        plt.savefig(f"{output_file}.png", dpi=DPI, bbox_inches='tight', transparent=True)
        print(f"Saved {output_file}.png")

        # Hide labels for clean SVG (Illustrator editing)
        hist_ylabel.set_visible(False)
        hist_xlabel.set_visible(False)
        bar_ylabel.set_visible(False)
        for lbl in bar_category_labels:
            lbl.set_visible(False)
        for lbl in condition_labels:
            lbl.set_visible(False)
        ax_hist.set_xticklabels([])
        ax_hist.set_yticklabels([])
        ax_bar.set_xticklabels([])
        ax_bar.set_yticklabels([])

        plt.savefig(f"{output_file}.svg", dpi=DPI, bbox_inches='tight', transparent=True)
        print(f"Saved {output_file}.svg")
        plt.close()

    print("\nGenerating stacked plot...", flush=True)
    create_stacked_plot(output_file=OUTPUT_FILE, xlim=XLIM)
    print("\n=== Plot generated successfully! ===", flush=True)
