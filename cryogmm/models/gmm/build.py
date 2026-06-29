"""
GMM Build Utilities

Helper functions for fitting per-cluster local PCA whiteners and Ledoit-Wolf
covariance matrices. Extracted from the pipeline build script so they can be
reused or tested independently.
"""

import torch
import numpy as np
from sklearn.covariance import ledoit_wolf

from cryogmm.preprocess.whiten import Whitener
from cryogmm.utils import assert_tensor, assert_numpy


# Default thresholds used throughout the pipeline
VARIANCE_THRESHOLD       = 0.95    # cumulative explained variance cutoff
EIGENVALUE_ABS_THRESHOLD = 1e-3    # nm²; PCA components below this are excluded
SMALL_CLUSTER_SIGMA_VAR  = 1e-6    # nm²; fallback variance for degenerate clusters


def compute_keepdims(
    X_k,
    variance_threshold=VARIANCE_THRESHOLD,
    eigenvalue_abs_threshold=EIGENVALUE_ABS_THRESHOLD,
):
    """Return the number of PCA components to keep for cluster data X_k.

    Uses two criteria and takes the more restrictive (smaller) value:
      k_var: components needed to explain `variance_threshold` cumulative variance.
      k_abs: components whose eigenvalue exceeds `eigenvalue_abs_threshold`.
    Result is additionally capped at n_positive (rank of sample covariance).

    Args:
        X_k (np.ndarray): Shape (N_samples, D_flat). Cluster coordinates, flattened.
        variance_threshold (float): Minimum cumulative variance to retain (default 0.95).
        eigenvalue_abs_threshold (float): Minimum absolute eigenvalue to keep (default 1e-3 nm²).

    Returns:
        int: Number of PCA dimensions to keep (at least 1).
    """
    X_centered = X_k - X_k.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.maximum(eigvals, 0.0)

    n_positive = int((eigvals > 1e-10).sum())

    cumvar = np.cumsum(eigvals) / (eigvals.sum() + 1e-12)
    k_var = int(np.searchsorted(cumvar, variance_threshold)) + 2
    k_abs = int((eigvals >= eigenvalue_abs_threshold).sum())

    keepdims = min(k_var, k_abs, n_positive)
    return max(keepdims, 1)


def build_cluster_whitener_and_params(
    k,
    xyz,
    cluster_id,
    center_id,
    cov_reg_min,
    device,
    variance_threshold=VARIANCE_THRESHOLD,
    eigenvalue_abs_threshold=EIGENVALUE_ABS_THRESHOLD,
    small_cluster_sigma_var=SMALL_CLUSTER_SIGMA_VAR,
):
    """Fit a local PCA whitener and Ledoit-Wolf covariance for cluster k.

    Degeneracy threshold: N >= 2*keepdims. Ledoit-Wolf is a shrinkage estimator
    and far less data-hungry than plain MLE covariance. Falls back to an identity
    whitener centered at the cluster center only for genuinely tiny clusters.

    Args:
        k (int): Cluster index.
        xyz (np.ndarray): Shape (N_total, N_atoms, 3). Aligned backbone coordinates.
        cluster_id (np.ndarray): Shape (N_total,). Cluster label per frame.
        center_id (np.ndarray): Shape (N_clusters,). Frame index of each cluster center.
        cov_reg_min (float): Minimum eigenvalue floor for Ledoit-Wolf covariance.
        device (str): Torch device string.
        variance_threshold (float): Passed to compute_keepdims.
        eigenvalue_abs_threshold (float): Passed to compute_keepdims.
        small_cluster_sigma_var (float): Variance for identity fallback covariance.

    Returns:
        tuple: (whitener, mu_k, Sigma_k, is_degenerate)
            - whitener (Whitener): Fitted Whitener for this cluster.
            - mu_k (torch.Tensor): Mean in whitened space, shape (M_k,).
            - Sigma_k (torch.Tensor): Covariance in whitened space, shape (M_k, M_k).
            - is_degenerate (bool): True if identity whitener was used.
    """
    mask = cluster_id == k
    Nk = int(mask.sum())
    D_in = xyz.shape[1] * xyz.shape[2]

    X_k = xyz[mask].reshape(Nk, -1)

    if Nk <= 2:
        keepdims_k = 1
        min_samples_needed = 3
    else:
        keepdims_k = compute_keepdims(X_k, variance_threshold, eigenvalue_abs_threshold)
        min_samples_needed = max(2 * keepdims_k, 3)

    if Nk < min_samples_needed:
        center_flat = xyz[center_id[k]].reshape(-1).astype(np.float32)
        D_whitener_dict = {
            "dim_in": D_in,
            "dim_out": D_in,
            "keepdims": D_in,
            "dim_cart_signal": None,
            "scale": False,
            "X0mean": center_flat,
            "Twhiten": np.eye(D_in, dtype=np.float32),
            "Tblacken": np.eye(D_in, dtype=np.float32),
            "std": np.ones(D_in, dtype=np.float32),
        }
        wtner_k = Whitener.from_dict(D_whitener_dict, device=device)
        mu_k = torch.zeros(D_in, dtype=torch.float32, device=device)
        Sigma_k = (
            torch.eye(D_in, dtype=torch.float32, device=device) * small_cluster_sigma_var
        )
        print(
            f"  Cluster {k}: N={Nk}, keepdims={keepdims_k}, need>={min_samples_needed} — "
            f"identity whitener at cluster center (bond filter skipped)",
            flush=True,
        )
        return wtner_k, mu_k, Sigma_k, True

    print(f"  Cluster {k}: N={Nk}, keepdims={keepdims_k}", flush=True)

    wtner_k = Whitener(X_k, keepdims=keepdims_k, scale=False, device=device)
    X_k_tensor = assert_tensor(X_k, arr_type=torch.float32, device=device)
    X_normed_k = wtner_k.whiten(X_k_tensor)

    Sigma_k_np, _ = ledoit_wolf(assert_numpy(X_normed_k).copy())
    Sigma_k_np = Sigma_k_np.astype(np.float32)
    eigvals, eigvecs = np.linalg.eigh(Sigma_k_np)
    eigvals = np.maximum(eigvals, cov_reg_min)
    Sigma_k_np = (eigvecs * eigvals) @ eigvecs.T
    Sigma_k_np = 0.5 * (Sigma_k_np + Sigma_k_np.T)
    Sigma_k = torch.tensor(Sigma_k_np, dtype=torch.float32, device=device)
    mu_k = X_normed_k.mean(0)

    return wtner_k, mu_k, Sigma_k, False
