# cryoGMM

Gaussian Mixture Model framework for conformational density approximation informed by cryo-EM images.

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `torch`, `mdtraj`, `scikit-learn`, `fpsample`, `tqdm`, `matplotlib`.

## Pipeline

Two tutorials, covering the steps before and after the cryo-EM likelihood
calculation:

- **[docs/fps_clustering_tutorial.md](docs/fps_clustering_tutorial.md)** —
  conformational subset construction: reduce a prior ensemble to a
  representative subset by farthest-point sampling in CV space.
- **[docs/tutorial.md](docs/tutorial.md)** — from building the GMM over that
  subset through to the figures.

```
CG / MD trajectory
        │
        ▼
  scripts/fps_clustering/fps_clustering.py     representative subset + clusters
        │
        ▼
  cryoLike / cryoSBI                           image-to-structure likelihood
  cryogmm.utils.multiplicative_gradient        per-cluster weights
        │
        ▼
  Step 0: scripts/step0_build_gmm/gmm_build.py
  Step 1: scripts/step1_compute_cvs/gmm_cvs.py
  Step 2: scripts/step2_make_figures/
```

## Library API

```python
from cryogmm import (
    Whitener,
    sample_gmm_with_bond_filter,
    compute_keepdims,
    build_cluster_whitener_and_params,
)
from cryogmm.preprocess import (
    dist, angle,
    fps_clustering, assign_clusters, resample_cluster_members,
    save_cluster_assignment, plot_fps_clustering,
)
from cryogmm.utils import (
    assert_tensor, assert_numpy, save_samples_to_pdb, load_aligned_trajectory,
)
```

See [docs/tutorial.md](docs/tutorial.md) for usage examples.
