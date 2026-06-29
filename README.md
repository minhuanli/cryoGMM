# cryoGMM

Gaussian Mixture Model framework for conformational density approximation informed by cryo-EM images.

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `torch`, `mdtraj`, `scikit-learn`, `tqdm`, `matplotlib`.

## Pipeline

The full analysis pipeline — from building the GMM to generating figures — is documented in **[docs/tutorial.md](docs/tutorial.md)**.

```
MD trajectory + cluster assignments + SBI weights
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
from cryogmm.preprocess import dist, angle
from cryogmm.utils import assert_tensor, assert_numpy, save_samples_to_pdb
```

See [docs/tutorial.md](docs/tutorial.md) for usage examples.
