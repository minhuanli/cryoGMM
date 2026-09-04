# Conformational Subset Construction Tutorial

This tutorial covers the **first** step of the pipeline: reducing a large prior
ensemble (a coarse-grained MD trajectory, or a pool of sampled conformations) to
a compact, representative subset by farthest-point sampling (FPS) in
collective-variable space.

It runs once, before any cryo-EM likelihood or reweighting calculation. Its
outputs feed two later stages:

- the **representative conformations** (`center_{k}.pdb`) are what the
  likelihood step scores against the experimental particle images;
- the **cluster assignment** (`center_idx.npy`, `cluster_labels.npy`) is what
  the GMM build step reads to estimate a local covariance around each center.

For everything downstream of this — building the GMM, computing CVs of the
sampled ensemble, and making figures — see
**[docs/tutorial.md](tutorial.md)**.

## Overview

```
CG / MD trajectory  (N_frames × N_atoms × 3)
        │
        ▼
  cryogmm-fps
  ─────────────────────────────────────────────
  Superpose all frames onto frame 0
  Embed each frame in a low-D CV space   → X  (N_frames, D_cv)
  Farthest-point sampling                → K maximally spread centers
  Nearest-center assignment              → cluster label per frame
  Repeat over seeds                      → independent "sets"
        │
        ▼  {output_root}/{K}_clusters/set_{i}/
        │      center_idx.npy, cluster_labels.npy, center_{k}.pdb
        │
        ├────────────────►  cryo-EM likelihood (cryoLike / cryoSBI)
        │                   scores center_{k}.pdb against the particles
        │
        └────────────────►  scripts/step0_build_gmm/gmm_build.py
                            --cluster_root {output_root}/{K}_clusters
```

---

## Installation

Same as the rest of the package (see
[docs/tutorial.md § Installation](tutorial.md#installation)):

```bash
git clone https://github.com/minhuanli/cryoGMM.git
cd cryoGMM
pip install -e .
```

This registers the `cryogmm-fps` command, which can be run from any directory.

FPS itself is delegated to [`fpsample`](https://github.com/leonardodalinky/fpsample),
a fast Rust implementation, installed as a dependency. A pure-numpy backend is
available with `--backend numpy` if `fpsample` is unavailable.

---

## Data Prerequisites

| File | Format | Description |
|------|--------|-------------|
| Trajectory | `.pt` (PyTorch) | Shape `(N_frames, N_atoms, 3)`, **nm units**. Use `--angstrom_to_nm` if stored in Å. Any MDTraj-readable trajectory file also works. |
| Topology | `.pdb` | MDTraj-readable topology matching the trajectory atom order. |
| Collective variables | computed on the fly, or `.dat` / `.npy` | See below. |

The trajectory should already sample the conformational space you care about —
FPS selects from what is there, it cannot extrapolate.

---

## Choosing collective variables

FPS is performed in a low-dimensional CV space, not in Cartesian or RMSD space.
Two or three CVs that separate the conformational states of interest work well;
the goal is a subset that covers the populated regions evenly rather than in
proportion to their prior population, so that rare states still get a
representative.

The script computes CVs directly from the aligned trajectory. Each `--cv` flag
adds one dimension:

| Spec | Meaning |
|------|---------|
| `dist:i,j` | Distance between atoms `i` and `j` |
| `angle:i,j,k` | Angle at atom `j` |
| `torsion:i,j,k,l` | Dihedral angle |
| `file:path` | One column of text, read with `np.loadtxt` (e.g. an externally computed RMSD to a reference state) |
| `npy:path` | A 1-D `.npy` array |

Atom indices refer to the **aligned trajectory**, in the topology's atom order.
Alternatively, pass a precomputed `(N_frames, D_cv)` array with `--cv_path` and
skip `--cv` entirely.

> **CVs are compared on their raw scale.** FPS measures distance in CV space
> with a plain Euclidean metric, so a CV spanning a numerically wider range
> counts for proportionally more when the centers are chosen. Mixed units make
> this easy to miss: in the example above the RMSD is in Å, spanning roughly
> 4–50, while the G6-A81 distance is in nm, spanning roughly 0.5–9 — so the
> RMSD axis carries about five times the weight in the selection. That is what
> the published clustering did. If you want your CVs weighted equally, rescale
> the columns yourself (dividing each by its standard deviation, say) and pass
> the result with `--cv_path`.

> **Note.** FPS could in principle be run on pairwise RMSD instead of CVs, but
> that is not supported here: the RMSD matrix is quadratic in the number of
> frames, and CV space is what the downstream analysis is expressed in anyway.

---

## Step: Run the FPS clustering

`cryogmm-fps` (source: `cryogmm/cli/fps_clustering.py`)

### Key arguments

| Argument | Description |
|----------|-------------|
| `--traj_path` | Path to trajectory (`.pt`, or MDTraj-readable) |
| `--traj_top` | Path to topology `.pdb` |
| `--output_root` | Output root; results go to `{output_root}/{n_clusters}_clusters/` |
| `--alignment_selection` | MDTraj selection for superposition onto frame 0 |
| `--cv` | One CV per flag, repeatable (see table above) |
| `--cv_path` | Precomputed `(N_frames, D_cv)` `.npy`, instead of `--cv` |
| `--cv_labels` | Axis labels for the diagnostic plot |
| `--n_clusters` | Size of the representative subset (default: 40) |
| `--seeds` | Comma-separated RNG seeds, one per set (default: `42,12345,162,160,70`) |
| `--backend` | `fpsample` (default) or `numpy` |
| `--angstrom_to_nm` | Divide coordinates by 10 for Å-stored trajectories |
| `--mode` | `seeds` (default) or `resample`; see below |
| `--refine` | Optional max-min refinement pass |
| `--no_pdb` | Skip writing the representative PDB files |

### Example command

The P4-P6 Martini monomer, clustered on an RMSD to the closed state and one
inter-residue distance — the settings used in the paper:

```bash
cryogmm-fps \
    --traj_path            /data/traj/positions_all_traj.pt \
    --traj_top             /data/traj/top.pdb \
    --output_root          /data/clusters/my_system \
    --alignment_selection  "(resi > 108) and name BB2" \
    --cv                   file:/data/traj/rmsd_to_closed.dat \
    --cv                   dist:33,529 \
    --cv_labels            "RMSD to closed state" "CV2, dist(G6-A81)" \
    --n_clusters           40 \
    --seeds                42,12345,162,160,70 \
    --backend              numpy \
    --refine
```

Here `(resi > 108) and name BB2` aligns on the structurally rigid part of the
molecule, so the CVs measure genuine conformational change rather than overall
tumbling. Atoms 33 and 529 are the BB2 beads of G6 and A81; the RMSD to the
closed state is computed separately and read from a text file.

This is cheap — a few minutes for ~10⁵ frames, dominated by loading the
trajectory, and it needs no GPU.

### Output directory layout

```
{output_root}/
  40_clusters/
    set_0/
      center_idx.npy          # (K,)  frame index of each representative
      cluster_labels.npy      # (N,)  cluster label per frame
      center_0.pdb            # representative conformations
      ...
      center_39.pdb
      clustering.png          # CV space coloured by cluster, centers marked
    set_1/
    ...
```

`{output_root}/40_clusters` is exactly what
[`gmm_build.py`](tutorial.md#step-0-build-the-gmm) expects as `--cluster_root`;
it appends `set_{set_id}` itself.

### Checking the result

The script prints the cluster occupancy for each set:

```
Cluster occupancy: min 9, median 709, max 12786
  WARNING: 1 cluster(s) hold fewer than 10 frames; their covariance will be
  poorly determined.
```

A very small cluster carries too little data for a meaningful local covariance.
The GMM build step handles this gracefully — a cluster with fewer than
`2 × keepdims` frames falls back to an identity whitener at the cluster center
and is flagged in `gmm_cluster_{k}_is_degenerate.npy` — but many such clusters
means the subset is too fine for the trajectory. Reduce `--n_clusters`, or
choose CVs that spread the density more evenly.

Also open `clustering.png`: the centers should tile the populated region of CV
space, with no large occupied area left without a nearby center.

---

## Resampling mode

The five seeds in the default run give five *independent* clusterings, and the
spread of the final reweighted ensemble across them is the error bar on the
result. Sometimes you want to isolate a different source of variation: how much
does the answer depend on *which member* of each cluster is used as the
representative structure, holding the clustering itself fixed?

`--mode resample` does that. It runs FPS once, then draws `--n_sets` alternative
member frames from within each cluster:

```bash
cryogmm-fps \
    ...same arguments... \
    --mode    resample \
    --n_sets  5
```

```
{output_root}/
  40_clusters/
    center_idx.npy            # the shared clustering
    cluster_labels.npy
    set_0/
      sample_0.pdb            # alternative representative for cluster 0
      ...
      sample_idx.txt          # frame index of each
      sampling.png
```

The clustering now lives at the root rather than under `set_{i}`, so pass
`--shared_clusters` to `gmm_build.py`
(see [docs/tutorial.md § Shared cluster assignments](tutorial.md#shared-cluster-assignments)).
The per-set `sample_{k}.pdb` files are the structures to score in the likelihood
step, one complete alternative subset per set.

---

## Notes

### How many clusters?

`--n_clusters` trades resolution against statistics in both directions: more
centers describe the conformational density more finely, but each cluster then
holds fewer frames (a worse local covariance) and the likelihood step has more
structures to score against every particle image. 40 is a reasonable starting
point for a system like P4-P6 with ~10⁵ frames. Increase it only if the
occupancy stays healthy and the likelihood calculation remains affordable.

### Reproducibility

`fpsample` randomises only the starting frame, and draws it from the global
numpy RNG, so `--seeds` makes each set exactly reproducible. The `numpy` backend
behaves the same way. To reproduce a single set, pass just its seed:
`--seeds 42`.

The two backends implement the same greedy algorithm, but `fpsample` computes in
single precision, so they can select different frames where distances are nearly
tied. Fix one backend for a set of runs you intend to compare.

### Max-min refinement

Greedy FPS is order-dependent and can leave two centers closer together than
necessary. `--refine` runs sweeps of random candidate swaps, accepting any swap
that increases the minimum pairwise distance between centers, and stops as soon
as a full sweep finds no improvement.

It is off by default because of its cost — the candidate pool is `--pool_frac`
(default 0.1) of the trajectory per center per sweep — not because it is
negligible. On the P4-P6 RMSD/CV2 run above it moved between 0 and 9 of the 40
centers depending on the seed, so whether you enable it is part of the
clustering definition: keep it fixed across the sets you intend to compare.

### Ångström-scale trajectories

Pass `--angstrom_to_nm` for trajectories stored in Å, and remember to keep the
unit convention consistent in the later steps (`gmm_build.py` and `gmm_cvs.py`
take the same flag, and `--cov_reg_min` must be raised to `1e-2`; see the
"Ångström-scale trajectories" note in [docs/tutorial.md](tutorial.md)).

### Library API

The same operations are available as functions, which is often more convenient
when the CVs need custom code:

```python
import numpy as np
from cryogmm.utils import load_aligned_trajectory
from cryogmm.preprocess import (
    dist, fps_clustering, resample_cluster_members,
    save_cluster_assignment, plot_fps_clustering,
)

traj = load_aligned_trajectory(
    "positions_all_traj.pt", "top.pdb",
    alignment_selection="(resi > 108) and name BB2",
)
xyz = traj.xyz

# Embed every frame in a 2-D CV space
rmsd = np.loadtxt("rmsd_to_closed.dat")   # RMSD to the closed state
cv2 = dist(xyz[:, 33], xyz[:, 529])       # G6-A81
X = np.vstack([rmsd, cv2]).T

for i, seed in enumerate([42, 12345, 162, 160, 70]):
    centers_idx, labels = fps_clustering(
        X, n_clusters=40, seed=seed, backend="numpy", refine=True,
    )
    save_cluster_assignment(
        f"./40_clusters/set_{i}", centers_idx, labels,
        xyz=xyz, topology=traj.top,
    )
    plot_fps_clustering(
        X, centers_idx, labels,
        xlabel="RMSD to closed state", ylabel="CV2, dist(G6-A81)",
        filename=f"./40_clusters/set_{i}/clustering.png",
    )
```

| Function | Purpose |
|----------|---------|
| `fps_clustering(X, n_clusters, seed=..., backend=..., refine=...)` | FPS selection plus nearest-center assignment; returns `(centers_idx, labels)` |
| `farthest_point_sampling(X, n_clusters, start_idx=..., backend=...)` | The selection alone |
| `assign_clusters(X, centers_idx)` | Nearest-center labels for an existing set of centers |
| `refine_centers_max_min(X, centers_idx, ...)` | The max-min refinement pass |
| `resample_cluster_members(labels, n_samples=...)` | Alternative representatives per cluster |
| `save_cluster_assignment(dir, centers_idx, labels, xyz=..., topology=...)` | Write the layout `gmm_build.py` reads |
| `plot_fps_clustering(X, centers_idx, labels, ...)` | The CV-space diagnostic plot |
| `load_aligned_trajectory(path, top, alignment_selection=...)` | Load and superpose a trajectory |

---

## Next steps

1. Score `center_{k}.pdb` against the experimental particle images with
   [cryoLike](https://github.com/flatironinstitute/CryoLike) or
   [cryoSBI](https://github.com/flatironinstitute/cryoSBI) to get an
   image-to-structure log-likelihood matrix.
2. Turn that matrix into per-structure weights with `cryogmm-reweight` — see
   **[docs/reweighting_tutorial.md](reweighting_tutorial.md)**.
3. Build the GMM and sample the reweighted ensemble, passing
   `--cluster_root {output_root}/{n_clusters}_clusters` to `gmm_build.py` —
   see **[docs/tutorial.md](tutorial.md)**.
