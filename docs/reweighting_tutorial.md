# Ensemble Reweighting Tutorial

This tutorial covers the step between the cryo-EM likelihood calculation and
the GMM build: turning an image-to-structure log-likelihood matrix into the
mixture weights over conformations.

It sits here in the pipeline:

```
representative conformations          (docs/fps_clustering_tutorial.md)
        │
        ▼
  cryoLike / cryoSBI                  image-to-structure log-likelihood
        │                             (n_images, n_structures)
        ▼
  cryogmm-reweight                    ← this tutorial
        │
        ▼  weights.pt
        │
  gmm_build.py --weights_path_template ...   (docs/tutorial.md)
```

---

## The method

The likelihood matrix says how well each candidate structure explains each
particle image. What it does not say is how *populated* each structure is —
that is what reweighting recovers.

Treat the ensemble as a mixture whose components are the fixed candidate
structures and whose proportions are unknown. The multiplicative gradient
algorithm finds the proportions that maximize the likelihood of the observed
image set, and [early stopping provides entropic regularization](https://arxiv.org/abs/2609.01688). Each iteration scales every weight by the average, over images, of
how much that structure contributes to explaining each image relative to the
current mixture. This is the expectation-maximization update for mixture
proportions; writing it multiplicatively keeps the weights on the simplex for
free — they start uniform, stay non-negative, and stay normalized.

Iteration stops on a certificate rather than a fixed budget. The quantity
`max(grad) - 1` upper-bounds the gap between the current log-likelihood and
that of the optimal weights, so once it drops below `--tol` the answer is known
to be within `tol` of the best achievable. Typical log-likelihood matrices converge in tens to
hundreds of iterations.

> The algorithm in `cryogmm/utils/reweighting.py` was written by **Luke Evans**.
> The method as applied to cryo-EM is introduced in
> [this manuscript](https://www.nature.com/articles/s42003-026-09859-6).

---

## Input

| File | Format | Description |
|------|--------|-------------|
| Log-likelihood matrix | `.npy` or `.pt` | Shape `(n_images, n_structures)`. Entry `(i, j)` is the log-likelihood of generating image `i` from structure `j`. |

It must be a **log**-likelihood, and it must be finite — the CLI refuses NaN or
inf rather than silently producing meaningless weights. Additive constants per
image are harmless: the update is invariant to scaling each image's likelihood
row by a constant.

Columns need not all be conformations. When the sample contains several
compositional species, the extra columns are additional mixture components
(dimers, junk/noise) and are reweighted alongside the conformations — the
resulting vector is what `gmm_build.py` expects; see the "Weight file format
note" in [docs/tutorial.md](tutorial.md).

---

## Step: Run the reweighting

`cryogmm-reweight` (source: `cryogmm/cli/reweighting.py`)

```bash
cryogmm-reweight \
    --log_likelihood /data/likelihood/set_0/J1001/log_likelihood.npy \
    --output         /data/sbi/set_0/J1001/weights.pt
```

### Key arguments

| Argument | Description |
|----------|-------------|
| `--log_likelihood` | Input matrix, `.npy` or `.pt` |
| `--output` | Where to write the weights; `.pt` is what `gmm_build.py` reads, `.npy` also supported |
| `--tol` | Early-stopping tolerance on the duality gap (default: `1e-3`) |
| `--max_iterations` | Cap if the tolerance is never met (default: 10000) |
| `--stats_frequency` | With `--verbose`, print the loss every this many iterations (default: 100) |
| `--device` | `cpu` (default) or e.g. `cuda:0` |
| `--double` | Run in float64; helpful for small `--tol` or when many weights are near 0. |
| `--verbose` | Print the loss as iteration proceeds |

The run is cheap — the cost is one pass over the likelihood matrix per
iteration. A large matrix benefits from `--device cuda:0`.

### Output

```
Loaded log-likelihood: 5000 images x 40 structures
exiting!
#iterations at exit: 3
Weights written to /data/sbi/set_0/J1001/weights.pt
```

The weights are written as a 1-D tensor, one entry per column of the input
matrix, summing to 1. Run it once per job and per set, matching the
`--weights_path_template` that `gmm_build.py` will be given.

### Checking the result

How concentrated the weights came out is the quickest sanity check:

```python
import torch
w = torch.load("weights.pt", weights_only=True).double()
print("effective structures:", float(torch.exp(-(w[w > 0] * w[w > 0].log()).sum())))
print("carrying 90%:", int((torch.cumsum(w.sort(descending=True).values, 0) < 0.9).sum()) + 1)
```

The first is the perplexity of the weight distribution — how many structures
the ensemble effectively rests on. Collapsing towards 1 means a single
structure is explaining every image, which usually points to a subset too
coarse to describe the data, or an overconfident likelihood. Close to the full
number of structures means the images are barely discriminating between them,
and the reweighted ensemble will look much like the prior.

Run with `--verbose` to watch the loss; it cannot increase, so an increase
indicates a numerical problem — try `--double`.

---

## Library API

```python
import torch
from cryogmm.utils import multiplicative_gradient

log_likelihood = torch.load("log_likelihood.pt", weights_only=True)

weights, stats = multiplicative_gradient(
    log_likelihood,            # (n_images, n_structures)
    tol=1e-3,
    max_iterations=10000,
    VERBOSE=True,
    TRACK_WEIGHTS=False,       # True also records the weights as they evolve
)
```

Note that it returns a **tuple**: the weights and a dictionary holding the
loss trace (`stats["losses"]`, `stats["idx"]`, and `stats["weights"]` when
`TRACK_WEIGHTS=True`).

| Function | Purpose |
|----------|---------|
| `multiplicative_gradient(log_likelihood, tol=..., ...)` | The reweighting itself; returns `(weights, stats)` |
| `log_marginal_likelihood(weights, log_likelihood)` | The objective being maximised |
| `grad_log_prob(weights, log_likelihood)` | One multiplicative update factor, computed in log space |

The weights are initialised uniform — the most entropic starting point — and
inherit the device and dtype of `log_likelihood`, so a matrix already on the
GPU is reweighted there.

---

## Next steps

Pass the weights to `gmm_build.py` via `--weights_path_template`, and continue
with **[docs/tutorial.md](tutorial.md)**.
