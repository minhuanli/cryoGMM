# Pipeline Tutorial

This tutorial walks through the full cryoGMM analysis pipeline — from fitting a Gaussian Mixture Model (GMM) to a set of clustered MD trajectories informed by cryo-EM SBI weights, through computing collective variables (CVs), to generating publication figures.

It starts from an existing cluster assignment. To produce one — reducing a prior ensemble to a representative subset by farthest-point sampling — see **[docs/fps_clustering_tutorial.md](fps_clustering_tutorial.md)** first.

## Overview

```
MD trajectory + cluster assignments + SBI weights
        │
        ▼
  Step 0: gmm_build.py
  ─────────────────────────────────────────────
  Per-cluster local PCA whitener (adaptive dims)
  Ledoit-Wolf covariance estimation
  Bond-distance-filtered GMM samples
        │
        ▼  {output_root}/{K}_centers/{Set,Job}_*/
        │
  Step 1: gmm_cvs.py
  ─────────────────────────────────────────────
  Load whiteners + sample GMM for each job/set
  Compute 3 CVs per sample
  Aggregate weight statistics across sets
        │
        ▼  {output_root}/{K}_centers/gmm_cvs/
        │
  Step 2: figure scripts
  ─────────────────────────────────────────────
  gmm_figure_graph_flow.py  →  stacked distribution plot
  gmm_TVD_plot.py           →  TVD heatmap
  gmm_trend_plot.py         →  CV trend + dimer % plot
        │
        ▼  {output_root}/{K}_centers/*.png, *.svg
```

---

## Installation

Clone the repo and install in editable mode:

```bash
git clone https://github.com/minhuanli/cryoGMM.git
cd cryoGMM
pip install -e .
```

Dependencies installed automatically: `numpy`, `torch`, `mdtraj`, `scikit-learn`, `fpsample`, `tqdm`, `matplotlib`.

---

## Data Prerequisites

Before running the pipeline you need:

| File | Format | Description |
|------|--------|-------------|
| Trajectory | `.pt` (PyTorch) | Shape `(N_frames, N_atoms, 3)`, **nm units**. Use `--angstrom_to_nm` if stored in Å. |
| Topology | `.pdb` | MDTraj-readable topology matching the trajectory atom order. |
| Cluster assignments | `center_idx.npy` + `cluster_labels.npy` | One pair per set. `center_idx` has shape `(K,)` (frame index of each cluster center); `cluster_labels` has shape `(N_frames,)`. |
| SBI weights | `.pt` (PyTorch tensor) | Shape `(K + 4,)` per job per set. Layout: `weights[:K]` = per-cluster monomer mixing weights; `weights[K]` = dimer-1; `weights[K+1]` = dimer-2; `weights[K+2]` = noise/junk. |

### Weight file format note

The default scripts assume the **4-class** weight structure above (monomer clusters + dimer1 + dimer2 + noise). If your decoder produces a **3-class** structure (no noise class, shape `(K+2,)`), update the weight indexing in `gmm_cvs.py` accordingly (see the `weights_arr` section near the bottom of the script).

---

## Step 0: Build the GMM

`scripts/step0_build_gmm/gmm_build.py`

Fits per-cluster local PCA whiteners using Ledoit-Wolf covariance estimation, then draws bond-distance-filtered samples from the GMM for each job and set.

### Key arguments

| Argument | Description |
|----------|-------------|
| `--traj_path` | Path to trajectory `.pt` file |
| `--traj_top` | Path to topology `.pdb` |
| `--cluster_root` | Cluster directory. May contain `{n_clusters}` template. |
| `--weights_path_template` | Weight file template. Must contain `{set_id}` and `{job_id}`. May contain `{n_clusters}`. |
| `--output_root` | Output root directory |
| `--bb_selection` | MDTraj selection for backbone atoms (e.g. `"name BB2"`) |
| `--alignment_selection` | MDTraj selection for superposition |
| `--job_ids` | Comma-separated SBI job IDs |
| `--n_clusters` | Number of clusters (default: 40) |
| `--cov_reg_min` | Minimum covariance eigenvalue (default: `1e-4`; use `1e-2` for Å-scale) |
| `--force` | Recompute cached whiteners |

### Example command

```bash
python scripts/step0_build_gmm/gmm_build.py \
    --traj_path             /data/traj/positions.pt \
    --traj_top              /data/traj/top.pdb \
    --cluster_root          /data/clusters/{n_clusters}_clusters \
    --weights_path_template /data/sbi/set_{set_id}/J{job_id}/weights.pt \
    --output_root           /data/gmm_results/my_system \
    --bb_selection          "name BB2" \
    --alignment_selection   "name BB2" \
    --job_ids               1001,1002,1003,1004 \
    --n_clusters            40 \
    --device                cuda:0
```

### Output directory layout

```
{output_root}/
  bond_dist_gaussian/
    mu_bond.npy           # mean consecutive bond lengths (D_bond,)
    sigma_bond.npy        # std of bond lengths (D_bond,)
  40_centers/
    Set_0/
      whiteners/
        whitener_cluster_0.pkl  # Whitener object
        ...
      gmm_cluster_0_Mu_normed.pt
      gmm_cluster_0_Sigma.pt
      gmm_cluster_0_is_degenerate.npy
    Job_1001/
      Set_0/
        gmm_pi.pt               # normalised mixture weights
        gmm_samples/
          gmm_sample_BB2_000.pdb
          gmm_centers.txt        # cluster index per sample
```

### Running on a cluster (Slurm)

The outer set loop is the natural array dimension. With 5 sets:

```bash
#SBATCH --array=0-4
#SBATCH -p gpu --gres=gpu:1 --cpus-per-task=16

SET_IDS=${SLURM_ARRAY_TASK_ID}
python scripts/step0_build_gmm/gmm_build.py \
    ...all args... \
    --set_ids $SET_IDS
```

Whiteners for the same set are cached on first run and reused across jobs.

---

## Step 1: Compute CVs

`scripts/step1_compute_cvs/gmm_cvs.py`

Loads the per-cluster whiteners built in step 0, samples the GMM for each job/set, and computes three collective variables per sample:

| CV | Definition | Units |
|----|-----------|-------|
| CV1 | Distance between backbone residues 50 and 120 | nm |
| CV2 | Distance between backbone residues 6 and 81 | nm |
| CV3 | Angle at residue 99, between residues 123–99–49 | rad |

Residue indices are defined in `RESI_*` constants at the top of the script and correspond to the BB2/SUGR/C1′ coarse-grained backbone topology.

### Key arguments

| Argument | Description |
|----------|-------------|
| `--gmm_output_root` | `--output_root` from step 0 |
| `--cluster_root`, `--weights_path_template` | Same as step 0 |
| `--n_samples` | Max GMM samples per job/set; also number of MD reference frames (default: 20000) |
| `--angstrom_to_nm` | Divide trajectory and GMM sample coordinates by 10 (for Å-stored trajectories) |
| `--only_md` | Regenerate MD reference CVs only; skip GMM sampling |

### Example command

```bash
python scripts/step1_compute_cvs/gmm_cvs.py \
    --traj_path             /data/traj/positions.pt \
    --traj_top              /data/traj/top.pdb \
    --gmm_output_root       /data/gmm_results/my_system \
    --cluster_root          /data/clusters/{n_clusters}_clusters \
    --weights_path_template /data/sbi/set_{set_id}/J{job_id}/weights.pt \
    --bb_selection          "name BB2" \
    --alignment_selection   "name BB2" \
    --job_ids               1001,1002,1003,1004 \
    --n_clusters            40 \
    --device                cuda:0
```

### Output files

All saved under `{gmm_output_root}/40_centers/gmm_cvs/`:

| File | Shape | Description |
|------|-------|-------------|
| `cvs_md.npy` | `(N_SAMPLES, 3)` | MD reference CVs |
| `weights_md.npy` | `(4,)` | Uniform prior weights |
| `cvs_{job_id}.npy` | object array of `(N_per_set, 3)` | Per-set GMM CVs |
| `weights_means_{job_id}.npy` | `(1, 4)` | Mean of [monomer, dimer1, dimer2, noise] across sets |
| `weights_stds_{job_id}.npy` | `(1, 4)` | Std of the above |

---

## Step 2: Generate Figures

All three figure scripts read from `{root_dir}/{system_path}/{n_clusters}_centers/gmm_cvs/` and write PNG + SVG outputs one directory up.

### Common arguments

| Argument | Description |
|----------|-------------|
| `--root_dir` | Parent of `system_path` (same as `--output_root` from step 0) |
| `--system_path` | Relative subdirectory, e.g. `"martini/martini_fullcg_CV2_RMSD_FPS"` |
| `--n_clusters` | Number of clusters (default: 40) |
| `--cv_index` | 0 = dist(50,120), 1 = dist(6,81), 2 = angle(123,99,49) |

### 2a. Stacked flow figure

```bash
python scripts/step2_make_figures/gmm_figure_graph_flow.py \
    --root_dir    /data/gmm_results \
    --system_path my_system \
    --n_clusters  40 \
    --cv_index    0
```

Shows the CV histogram and discrete state probabilities for each experimental condition, stacked vertically. Saves `gmm_dist_graph_flow_my_system_N40_cv0.png` and `.svg`.

Optional: `--job_ids` and `--labels` to override the default condition ordering.

### 2b. TVD heatmap

```bash
python scripts/step2_make_figures/gmm_TVD_plot.py \
    --root_dir    /data/gmm_results \
    --system_path my_system \
    --n_clusters  40 \
    --cv_index    0
```

Computes pairwise Total Variation Distance between all condition pairs. Saves `tvd_heatmap_my_system_N40_cv0.png` and `.svg`.

### 2c. Trend plot

```bash
python scripts/step2_make_figures/gmm_trend_plot.py \
    --root_dir    /data/gmm_results \
    --system_path my_system \
    --n_clusters  40 \
    --cv_index    0 \
    --job_ids     1009,1028,1031,1005,1033,1032 \
    --x_values    0.0,0.5,1.0,3.0,5.0,10.0 \
    --xlabel      "Mg²⁺ Concentration (mM)"
```

Shows mean CV value and dimer percentage as a function of a quantitative condition. Saves `cv_trends_my_system_N40_cv0.png` and `.svg`.

### Running all figures at once

```bash
ROOT_DIR=/data/gmm_results SYSTEM_PATH=my_system N_CLUSTERS=40 \
    bash scripts/step2_make_figures/run_figures.sh
```

---

## Notes

### Ångström-scale trajectories

If your trajectory is stored in Å (e.g. from FarFar or all-atom simulations), pass `--angstrom_to_nm` to both `gmm_build.py` and `gmm_cvs.py`. Also increase `--cov_reg_min` to `1e-2` since eigenvalue thresholds scale quadratically with coordinate units.

### Degenerate clusters

Clusters with very few samples (< 2 × keepdims) receive an identity whitener centered at the cluster center. These clusters bypass the bond-distance filter entirely (flag saved in `gmm_cluster_k_is_degenerate.npy`). This is intentional — the cluster center is a real MD frame with valid bond lengths.

### Shared cluster assignments

If all sets share the same clustering (e.g. for resampling runs), pass `--shared_clusters` to `gmm_build.py`. This loads `center_idx.npy` and `cluster_labels.npy` directly from `--cluster_root` instead of `--cluster_root/set_{set_id}/`.

### Library API

The two helper functions promoted to the `cryogmm` package can be used directly:

```python
from cryogmm.models.gmm import compute_keepdims, build_cluster_whitener_and_params, sample_gmm_with_bond_filter

# Determine adaptive PCA dimensionality for a cluster
keepdims = compute_keepdims(X_k, variance_threshold=0.95)

# Fit whitener + Ledoit-Wolf covariance
whitener, mu, Sigma, is_degenerate = build_cluster_whitener_and_params(
    k, xyz, cluster_id, center_id, cov_reg_min=1e-4, device="cuda:0"
)

# Sample from the full GMM with bond filtering
samples_xyz, cluster_ids = sample_gmm_with_bond_filter(
    pi, means, covariances, whiteners, mu_bond, sigma_bond,
    bond_sigma_cutoff=4.0, n_atoms=n_atoms, num_samples=200,
    oversample_factor=20, device="cuda:0"
)
```
