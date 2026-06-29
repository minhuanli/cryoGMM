#!/bin/bash
# ============================================================================
# Example: Compute GMM CVs (Step 1)
#
# Runs gmm_cvs.py for a single system. Must be run after step 0 (gmm_build.py).
# All paths must match those used in the build step.
#
# Usage:
#   bash run_gmm_cvs.sh
#   ONLY_MD=1 bash run_gmm_cvs.sh    # regenerate MD reference CVs only
# ============================================================================

# --- Required: set these to match your step-0 configuration ---
TRAJ_PATH=""          # same trajectory used in step 0
TRAJ_TOP=""           # same topology used in step 0
GMM_OUTPUT_ROOT=""    # --output_root from step 0
CLUSTER_ROOT=""       # same cluster root used in step 0
WEIGHTS_TEMPLATE=""   # same weight template used in step 0
BB_SEL=""             # same backbone selection used in step 0
ALIGN_SEL=""          # same alignment selection used in step 0
JOB_IDS=""            # comma-separated SBI job IDs

# --- Optional tuning ---
N_CLUSTERS=${N_CLUSTERS:-40}
N_SAMPLES=${N_SAMPLES:-20000}
COV_REG_MIN=${COV_REG_MIN:-1e-4}
BOND_SIGMA_CUTOFF=${BOND_SIGMA_CUTOFF:-4.0}
BOND_OVERSAMPLE=${BOND_OVERSAMPLE:-20}
SET_IDS=${SET_IDS:-"0,1,2,3,4"}
DEVICE=${DEVICE:-"cuda:0"}

ANG_FLAG=""
if [[ "${ANGSTROM_TO_NM:-0}" == "1" ]]; then
    ANG_FLAG="--angstrom_to_nm"
fi

ONLY_MD_FLAG=""
if [[ "${ONLY_MD:-0}" == "1" ]]; then
    ONLY_MD_FLAG="--only_md"
fi

python "$(dirname "$0")/gmm_cvs.py" \
    --traj_path              "$TRAJ_PATH" \
    --traj_top               "$TRAJ_TOP" \
    --gmm_output_root        "$GMM_OUTPUT_ROOT" \
    --cluster_root           "$CLUSTER_ROOT" \
    --weights_path_template  "$WEIGHTS_TEMPLATE" \
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
    $ANG_FLAG \
    $ONLY_MD_FLAG
