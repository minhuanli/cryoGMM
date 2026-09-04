#!/bin/bash
# Conformational subset construction — FPS in CV space.
# Runs once on the prior ensemble, before the likelihood and reweighting steps.

DATA_ROOT=/path/to/martini_monomer_traj
OUTPUT_ROOT=/path/to/fps_clustering

cryogmm-fps \
    --traj_path            ${DATA_ROOT}/positions_all_traj.pt \
    --traj_top             ${DATA_ROOT}/top.pdb \
    --output_root          ${OUTPUT_ROOT} \
    --alignment_selection  "(resi > 108) and name BB2" \
    --cv                   file:${DATA_ROOT}/rmsd_to_closed.dat \
    --cv                   dist:33,529 \
    --cv_labels            "RMSD to closed state" "CV2, dist(G6-A81)" \
    --n_clusters           40 \
    --seeds                42,12345,162,160,70 \
    --backend              numpy \
    --refine
