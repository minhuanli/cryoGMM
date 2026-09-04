"""
Ensemble reweighting — multiplicative gradient on an image-to-structure
log-likelihood matrix.

Reads the log-likelihood matrix produced by the likelihood step (cryoLike or
cryoSBI), runs the multiplicative gradient algorithm to convergence, and writes
the resulting mixture weights in the layout the GMM build step reads.

The algorithm itself lives in `cryogmm/utils/reweighting.py` and was written by
Luke Evans; this is a thin command-line wrapper around it.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from cryogmm.utils import multiplicative_gradient


def load_matrix(path, device, dtype=torch.float32):
    """Load a log-likelihood matrix from .pt or .npy, as a 2-D torch tensor."""
    path = str(path)
    if path.endswith(".pt"):
        arr = torch.load(path, weights_only=True)
    elif path.endswith(".npy"):
        arr = torch.from_numpy(np.load(path))
    else:
        raise ValueError(
            f"Unsupported log-likelihood format '{path}'. Expected .pt or .npy."
        )
    arr = torch.as_tensor(arr, dtype=dtype, device=device)
    if arr.ndim != 2:
        raise ValueError(
            f"Log-likelihood must be 2-D (n_images, n_structures), got shape "
            f"{tuple(arr.shape)}"
        )
    return arr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensemble reweighting by the multiplicative gradient algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  cryogmm-reweight \\\n"
            "      --log_likelihood /data/likelihood/set_0/J1001/log_likelihood.npy \\\n"
            "      --output         /data/sbi/set_0/J1001/weights.pt\n"
        ),
    )
    parser.add_argument(
        "--log_likelihood", type=str, required=True,
        help="Image-to-structure log-likelihood matrix, shape "
             "(n_images, n_structures), as .pt or .npy.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Where to write the weights (.pt, as read by gmm_build.py; .npy "
             "also supported).",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-3,
        help="Early-stopping tolerance. Iteration stops once the duality gap "
             "falls below this, which bounds how far the log-likelihood is "
             "from optimal (default: 1e-3).",
    )
    parser.add_argument("--max_iterations", type=int, default=10000,
                        help="Iteration cap if the tolerance is never met "
                             "(default: 10000).")
    parser.add_argument("--stats_frequency", type=int, default=100,
                        help="With --verbose, print the loss every this many "
                             "iterations (default: 100).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device, e.g. cpu or cuda:0 (default: cpu).")
    parser.add_argument("--double", action="store_true",
                        help="Run in float64. Slower, but worth it for very "
                             "tight --tol.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print the loss as the iteration proceeds.")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float64 if args.double else torch.float32

    log_likelihood = load_matrix(args.log_likelihood, args.device, dtype)
    n_images, n_structures = log_likelihood.shape
    print(f"Loaded log-likelihood: {n_images} images x {n_structures} structures",
          flush=True)

    if not torch.isfinite(log_likelihood).all():
        raise ValueError(
            "Log-likelihood contains NaN or inf. Check the likelihood step's "
            "output before reweighting."
        )

    weights, _ = multiplicative_gradient(
        log_likelihood,
        tol=args.tol,
        max_iterations=args.max_iterations,
        stats_frequency=args.stats_frequency,
        VERBOSE=args.verbose,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights_cpu = weights.detach().cpu()
    if out_path.suffix == ".npy":
        np.save(out_path, weights_cpu.numpy())
    else:
        torch.save(weights_cpu, out_path)
    print(f"Weights written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
