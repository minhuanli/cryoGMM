#!/bin/bash
# ============================================================================
# Example: Generate All Figures (Step 2)
#
# Runs all three figure scripts for a single system and all three CV indices.
# Must be run after step 1 (gmm_cvs.py).
#
# Usage:
#   bash run_figures.sh
#   CV=1 bash run_figures.sh     # only CV index 1
# ============================================================================

# --- Required: set these to match your pipeline configuration ---
ROOT_DIR=""       # same as --output_root from step 0 (parent of SYSTEM_PATH)
SYSTEM_PATH=""    # relative path to system, e.g. "martini/martini_fullcg_CV2_RMSD_FPS"
N_CLUSTERS=${N_CLUSTERS:-40}

# Which CVs to plot: 0=dist(50,120), 1=dist(6,81), 2=angle(123,99,49)
CV=${CV:-"all"}   # set to 0, 1, or 2 to run a single CV

SCRIPT_DIR="$(dirname "$0")"

for CV_INDEX in 0 1 2; do
    if [[ "$CV" != "all" && "$CV" != "$CV_INDEX" ]]; then
        continue
    fi

    echo "=== CV index $CV_INDEX ==="

    echo "-- Flow figure --"
    python "$SCRIPT_DIR/gmm_figure_graph_flow.py" \
        --root_dir     "$ROOT_DIR" \
        --system_path  "$SYSTEM_PATH" \
        --n_clusters   "$N_CLUSTERS" \
        --cv_index     "$CV_INDEX"

    echo "-- TVD heatmap --"
    python "$SCRIPT_DIR/gmm_TVD_plot.py" \
        --root_dir     "$ROOT_DIR" \
        --system_path  "$SYSTEM_PATH" \
        --n_clusters   "$N_CLUSTERS" \
        --cv_index     "$CV_INDEX"

    echo "-- Trend plot --"
    python "$SCRIPT_DIR/gmm_trend_plot.py" \
        --root_dir     "$ROOT_DIR" \
        --system_path  "$SYSTEM_PATH" \
        --n_clusters   "$N_CLUSTERS" \
        --cv_index     "$CV_INDEX"
done

echo "=== All figures generated ==="
