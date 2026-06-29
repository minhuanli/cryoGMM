#!/bin/bash
# ============================================================================
# Example: Build GMM (Step 0)
#
# Runs gmm_build.py for a single system. Adjust variables below for your data.
# For cluster deployments, wrap in a SLURM array job over SET_IDS (0-4).
#
# Usage:
#   bash run_gmm_build.sh
#   FORCE=1 bash run_gmm_build.sh      # recompute cached whiteners
#   SET_ID=2 bash run_gmm_build.sh     # run a specific set only
# ============================================================================

# --- Required: set these for your system ---
TRAJ_PATH=""          # path to positions .pt file (N_frames, N_atoms, 3), nm units
TRAJ_TOP=""           # path to topology .pdb file
CLUSTER_ROOT=""       # path to cluster directory (may contain {n_clusters})
WEIGHTS_TEMPLATE=""   # weight file template, e.g. "/path/set_{set_id}/J{job_id}/weights.pt"
OUTPUT_ROOT=""        # output root directory
BB_SEL=""             # MDTraj selection for backbone atoms, e.g. "name BB2"
ALIGN_SEL=""          # MDTraj selection for superposition, e.g. "name BB2"
JOB_IDS=""            # comma-separated SBI job IDs, e.g. "1001,1002,1003"

# --- Optional tuning ---
N_CLUSTERS=${N_CLUSTERS:-40}
N_SAMPLES=${N_SAMPLES:-200}
COV_REG_MIN=${COV_REG_MIN:-1e-4}   # use 1e-2 for Angstrom-scale coordinates
BOND_SIGMA_CUTOFF=${BOND_SIGMA_CUTOFF:-4.0}
BOND_OVERSAMPLE=${BOND_OVERSAMPLE:-20}
SET_IDS=${SET_IDS:-"0,1,2,3,4"}
DEVICE=${DEVICE:-"cuda:0"}

FORCE_FLAG=""
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG="--force"
fi

python "$(dirname "$0")/gmm_build.py" \
    --traj_path              "$TRAJ_PATH" \
    --traj_top               "$TRAJ_TOP" \
    --cluster_root           "$CLUSTER_ROOT" \
    --weights_path_template  "$WEIGHTS_TEMPLATE" \
    --output_root            "$OUTPUT_ROOT" \
    --bb_selection           "$BB_SEL" \
    --alignment_selection    "$ALIGN_SEL" \
    --job_ids                "$JOB_IDS" \
    --set_ids                "$SET_IDS" \
    --n_clusters             "$N_CLUSTERS" \
    --n_samples              "$N_SAMPLES" \
    --cov_reg_min            "$COV_REG_MIN" \
    --bond_sigma_cutoff      "$BOND_SIGMA_CUTOFF" \
    --bond_oversample_factor "$BOND_OVERSAMPLE" \
    --device                 "$DEVICE" \
    $FORCE_FLAG
