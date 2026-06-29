import time
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal


def stable_logdet(matrix, epsilon=1e-7):
    # Compute eigenvalues
    eigvals = torch.linalg.eigvalsh(matrix)

    # Clamp eigenvalues to ensure they're positive
    eigvals = torch.clamp(eigvals, min=epsilon)

    # Log determinant is the sum of the log of eigenvalues
    return torch.sum(torch.log(eigvals), dim=-1)


def batched_quad_term(diff, Sigma_inv, batch_size=20000):
    """
    Computes the quadratic term in batches over `n` to reduce memory usage.

    Args:
        diff (torch.Tensor): Tensor of shape (n, k, m)
        Sigma_inv (torch.Tensor): Tensor of shape (k, m, d)
        batch_size (int): The batch size to process at a time

    Returns:
        torch.Tensor: Computed quadratic terms of shape (n, k)
    """
    n, k, m = diff.shape
    d = Sigma_inv.shape[2]

    # Initialize result storage
    quad_terms = torch.empty(n, k, device=diff.device, dtype=diff.dtype)

    # Process in batches
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)

        # Slice batch
        diff_batch = diff[i:end_i]  # (batch_size, k, m)

        quad_terms[i:end_i] = torch.einsum(
            "nkm,kmd,nkd->nk", diff_batch, Sigma_inv, diff_batch
        )

    return quad_terms


def batched_weighted_outer_product(diff, responsibilities, N_k, batch_size=20000):
    """
    Computes the weighted sum of outer products in batches over `n` to reduce memory usage.

    Args:
        diff (torch.Tensor): Tensor of shape (n, k, m)
        responsibilities (torch.Tensor): Tensor of shape (n, k)
        N_k (torch.Tensor): Tensor of shape (k,) containing normalizing factors
        batch_size (int): The batch size to process at a time

    Returns:
        torch.Tensor: Computed covariance-like term of shape (k, m, m)
    """
    n, k, m = diff.shape

    # Initialize the accumulation tensor
    Sigmas_temp = torch.zeros(k, m, m, device=diff.device, dtype=diff.dtype)

    # Process in batches
    for i in range(0, n, batch_size):
        end_i = min(i + batch_size, n)

        # Slice batch
        diff_batch = diff[i:end_i]  # [batch_size, K, M]
        resp_batch = responsibilities[i:end_i]  # [batch_size, K]

        # Compute weighted differences: [batch_size, K, M]
        diff_weighted = diff_batch * resp_batch[:, :, None]

        # Compute batch outer product sum: [K, M, M]
        Sigmas_temp += torch.einsum("nkm,nkd->kmd", diff_weighted, diff_batch)

    # Normalize by N_k (element-wise division)
    Sigmas_temp /= N_k[:, None, None]

    return Sigmas_temp


def batched_gmm_sample_with_clusters(gmm: MixtureSameFamily, num_samples, batch_size=100):
    """
    Samples from a MixtureSameFamily distribution in batches for memory efficiency, returning both the samples
    and their corresponding component assignments.

    Args:
        gmm (MixtureSameFamily): The GMM distribution.
        num_samples (int): The total number of samples needed.
        batch_size (int): The number of samples to generate per batch.

    Returns:
        tuple(torch.Tensor, torch.Tensor):
            - samples of shape (num_samples, M)
            - cluster indices of shape (num_samples,)
    """
    mix_probs = gmm.mixture_distribution.probs  # Mixture weights (K,)
    component_distribution = (
        gmm.component_distribution
    )  # The underlying distribution (K, M)

    samples = []
    cluster_indices = []
    counts = 0
    for _ in range(0, num_samples, batch_size):
        batch_size_actual = min(
            batch_size, num_samples - counts
        )  # Handle last batch

        # Step 1: Sample cluster indices based on the mixture probabilities
        cluster_idx = torch.multinomial(mix_probs, batch_size_actual, replacement=True)

        # Step 2: Sample from the chosen components
        batch_samples = component_distribution.sample(
            (batch_size_actual,)
        )  # Shape (batch_size, K, M)
        selected_samples = batch_samples[
            torch.arange(batch_size_actual), cluster_idx
        ]  # Select correct components

        samples.append(selected_samples)
        cluster_indices.append(cluster_idx)
        counts += batch_size_actual

    assert counts == num_samples, "Sample count mismatch"

    return torch.cat(samples, dim=0), torch.cat(cluster_indices, dim=0)

def sample_gmm_with_bond_filter(
    pi,
    means,
    covariances,
    whiteners,
    mu_bond,
    sigma_bond,
    bond_sigma_cutoff,
    n_atoms,
    num_samples,
    oversample_factor,
    device,
    is_degenerate=None,
    cov_reg_min=1e-4,
):
    """
    Sample from a GMM with per-cluster local PCA whiteners and bond-distance filtering.

    Each cluster k has its own latent space of potentially different dimensionality M_k.
    Samples are proposed by oversampling from the per-cluster Gaussian, projecting back to
    Cartesian coordinates via the cluster whitener, then filtering based on backbone bond
    length plausibility.

    Args:
        pi (torch.Tensor): Mixture weights of shape (K,) on device.
        means (list[torch.Tensor]): Per-cluster mean vectors, each shape (M_k,) on device.
        covariances (list[torch.Tensor]): Per-cluster covariance matrices, each shape
            (M_k, M_k) on device.
        whiteners (list[Whitener]): Per-cluster Whitener objects.
        mu_bond (np.ndarray): Mean consecutive backbone bond lengths, shape (D_bond,).
        sigma_bond (np.ndarray): Std of backbone bond lengths, shape (D_bond,).
        bond_sigma_cutoff (float): Cutoff width w; samples with mean bond deviation > w*sigma
            are rejected (log_p_cutoff = log_p_max - w**2 * D_bond / 2).
        n_atoms (int): Number of backbone atoms (used to reshape flat coords to (N_atoms, 3)).
        num_samples (int): Total number of samples to return.
        oversample_factor (int): How many times more samples to propose per cluster before
            filtering. Increase if many proposals are rejected.
        device (str): Torch device string.
        is_degenerate (list[bool] or None): Per-cluster flag. If True for cluster k, the bond
            filter is skipped and all proposed samples are accepted directly (used for clusters
            that use an identity whitener pinned to the cluster center). If None, filtering is
            applied to all clusters.
        cov_reg_min (float): Minimum eigenvalue regularization for per-cluster covariances.

    Returns:
        tuple:
            - gmm_samples_xyz (np.ndarray): Filtered samples, shape (num_samples, n_atoms, 3).
            - gmm_ids (torch.Tensor): Cluster index for each sample, shape (num_samples,).
    """
    K = len(means)
    if is_degenerate is None:
        is_degenerate = [False] * K

    # Build bond distance Gaussian (shared across clusters)
    mu_bond_t = torch.tensor(mu_bond, dtype=torch.float32, device=device)
    cov_bond = torch.diag(torch.tensor(sigma_bond ** 2, dtype=torch.float32, device=device))
    bond_mvn = MultivariateNormal(mu_bond_t, cov_bond)
    D_bond = len(mu_bond)
    log_p_max = bond_mvn.log_prob(mu_bond_t)
    log_p_cutoff = log_p_max - bond_sigma_cutoff ** 2 * D_bond / 2

    # Draw all cluster assignments at once from the mixture weights
    cluster_ids_all = torch.multinomial(pi, num_samples, replacement=True)  # (num_samples,)
    counts_k = torch.bincount(cluster_ids_all, minlength=K)  # (K,)

    all_xyz = []
    all_ids = []

    for k in range(K):
        n_k = int(counts_k[k].item())
        if n_k == 0:
            continue

        mu_k = means[k]
        Sigma_k = covariances[k]
        wtner_k = whiteners[k]

        # Regularize covariance before sampling to ensure positive definiteness
        Sigma_k_reg = regularize_covariance(Sigma_k, max=None, min=cov_reg_min)
        gauss_k = MultivariateNormal(mu_k, Sigma_k_reg)

        # Clamp proposed count: at least 1000 for a decent filter pool, at most 20000
        N_MIN, N_MAX = 1000, 20000
        n_proposed_base = int(np.clip(n_k * oversample_factor, N_MIN, N_MAX))

        if is_degenerate[k]:
            # Cluster center is a real MD frame — bond lengths are guaranteed valid.
            # Skip filter and take all proposed samples directly.
            z_proposed = gauss_k.sample((n_proposed_base,))
            xyz_flat = wtner_k.blacken(z_proposed).cpu().numpy()
            xyz_3d = xyz_flat.reshape(n_proposed_base, n_atoms, 3)
            xyz_filtered = xyz_3d
            n_filtered = n_proposed_base
        else:
            # Adaptive retry: double proposals up to N_MAX if nothing passes the filter.
            # Always runs at least once (attempt_factors is non-empty).
            n_filtered = 0
            n_proposed_last = 0
            xyz_3d = np.empty((0, n_atoms, 3), dtype=np.float32)
            log_p = torch.empty(0, device=device)
            xyz_filtered = xyz_3d  # will be reassigned in loop

            for attempt_factor in [1, 2, 4]:
                n_proposed = int(np.clip(n_proposed_base * attempt_factor, N_MIN, N_MAX))
                if n_proposed == n_proposed_last:
                    break  # already at N_MAX, no point retrying
                n_proposed_last = n_proposed

                z_proposed = gauss_k.sample((n_proposed,))
                xyz_flat = wtner_k.blacken(z_proposed).cpu().numpy()
                xyz_3d = xyz_flat.reshape(n_proposed, n_atoms, 3)
                d_bonds = np.linalg.norm(xyz_3d[:, 1:] - xyz_3d[:, :-1], axis=2)
                log_p = bond_mvn.log_prob(
                    torch.tensor(d_bonds, dtype=torch.float32, device=device)
                )
                mask = (log_p > log_p_cutoff).cpu().numpy()
                xyz_filtered = xyz_3d[mask]
                n_filtered = int(mask.sum())

                if n_filtered > 0:
                    break
                print(
                    f"  Cluster {k}: {n_proposed} proposals, 0 passed bond filter "
                    f"(best log_p={log_p.max().item():.1f}, cutoff={log_p_cutoff.item():.1f}). "
                    f"Retrying with more proposals...",
                    flush=True,
                )

            if n_filtered == 0:
                # All retries exhausted — soft fallback: take the n_k proposals with
                # the highest bond log-prob regardless of the cutoff.
                log_p_np = log_p.cpu().numpy()
                print(
                    f"Warning: cluster {k}: 0/{n_proposed_last} samples passed bond filter "
                    f"after all retries (best={log_p_np.max():.1f}, cutoff={log_p_cutoff.item():.1f}). "
                    f"Falling back to top-{int(n_k)} by bond log-prob.",
                    flush=True,
                )
                top_indices = np.argsort(log_p_np)[-int(n_k):]
                all_xyz.append(xyz_3d[top_indices])
                all_ids.append(torch.full((int(n_k),), k, dtype=torch.long))
                continue  # skip normal index selection below

            if n_filtered < n_k:
                print(
                    f"Warning: cluster {k} produced only {n_filtered}/{int(n_k)} samples after bond "
                    f"filter. Sampling with replacement.",
                    flush=True,
                )

        if n_filtered < n_k:
            indices = np.random.choice(n_filtered, n_k, replace=True)
        else:
            indices = np.random.choice(n_filtered, n_k, replace=False)

        all_xyz.append(xyz_filtered[indices])
        all_ids.append(torch.full((n_k,), k, dtype=torch.long))

    gmm_samples_xyz = np.concatenate(all_xyz, axis=0)   # (num_samples, n_atoms, 3)
    gmm_ids = torch.cat(all_ids, dim=0)                 # (num_samples,)

    return gmm_samples_xyz, gmm_ids


def infer_gmm_sigmas(Sigmas_init, diff, pi, N_iter=100, eps=1e-5):
    nll_track = []
    Sigmas = Sigmas_init.clone()
    pbar = tqdm(range(N_iter), desc="Iterations")
    for i in pbar:
        Sigma_reg = Sigmas + torch.eye(Sigmas.shape[-1]).to(Sigmas) * 1e-5
        Sigma_inv = torch.linalg.inv(Sigma_reg)  # [K, M, M]
        # Calculate the responsibility
        quad_term = batched_quad_term(diff, Sigma_inv, batch_size=20000)# [N, K]
        log_pdf = -0.5 * (
            quad_term + M * np.log(2.0 * torch.pi) + stable_logdet(Sigma_reg)
        )  # Compute log of Gaussian densities, [N, K]
        log_pdf += torch.log(pi)  # Add log cluster weights, [N, K]
        responsibilities = torch.softmax(log_pdf, dim=1)
        NLL = -torch.logsumexp(log_pdf, dim=1).sum()

        # Use resiponsibility to update covariance matrix
        N_k = responsibilities.sum(0)  # [K]
        # Compute weighted sum of outer products in a memory-efficient way
        Sigmas_temp = batched_weighted_outer_product(diff, responsibilities, N_k, batch_size=20000) # [K, M, M]
        Sigmas = Sigmas_temp.clone()
        nll_track.append(NLL.item())
        pbar.set_postfix(NLL=f"{NLL.item():.3e}")
        if NLL.item() is np.nan:
            print("NaN detected in NLL. Exiting...", flush=True)
            break
    return Sigmas, nll_track

# Define the GMM using MixtureSameFamily
def create_gmm(weights, means, covariances):
    """
    Creates a Gaussian Mixture Model (GMM) using MixtureSameFamily in PyTorch.

    Args:
        weights (Tensor): Mixture weights [K]
        means (Tensor): Mean vectors [K, M]
        covariances (Tensor): Covariance matrices [K, M, M]

    Returns:
        MixtureSameFamily: A PyTorch GMM distribution object
    """
    # Step 1: Define categorical distribution for cluster selection
    mixture_distribution = Categorical(weights)

    # Step 2: Define multivariate normal components
    component_distribution = MultivariateNormal(means, covariances)

    # Step 3: Construct the MixtureSameFamily distribution
    gmm = MixtureSameFamily(mixture_distribution, component_distribution)

    return gmm


def load_gmm(weights_path, means_path, covariance_path, device="cuda:0"):
    """
    Load a Gaussian Mixture Model (GMM) from saved files.

    Args:
        weights_path (str): Path to the weights file
        means_path (str): Path to the means file
        covariance_path (str): Path to the covariance file

    Returns:
        MixtureSameFamily: A PyTorch GMM distribution object
    """
    if type(weights_path) is str:
        weights = torch.load(weights_path, weights_only=True).to(device)
    else:
        weights = weights_path.to(device)

    if type(means_path) is str:
        means = torch.load(means_path, weights_only=True).to(device)
    else:
        means = means_path.to(device)

    if type(covariance_path) is str:
        covariances = torch.load(covariance_path, weights_only=True).to(device)
    else:
        covariances = covariance_path.to(device)

    return create_gmm(weights, means, covariances)


def regularize_covariance(cov_matrix: torch.Tensor, max: float = 1e0, min: float = 1e-4) -> torch.Tensor:
    """
    Regularizes a covariance matrix by clipping its small eigenvalues.

    Args:
        cov_matrix (torch.Tensor): The input covariance matrix (must be symmetric).
        epsilon (float): The minimum value for the eigenvalues.

    Returns:
        torch.Tensor: The regularized covariance matrix.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    eigenvalues_clipped = torch.clamp(eigenvalues, max=max, min=min)

    cov_matrix_regularized = eigenvectors @ torch.diag_embed(eigenvalues_clipped) @ eigenvectors.mT

    # Float32 Q @ diag(λ) @ Q.T accumulates roundoff that breaks bit-symmetry.
    # MultivariateNormal's PD check (~1e-6 symmetry tolerance) rejects this, so symmetrize.
    cov_matrix_regularized = 0.5 * (cov_matrix_regularized + cov_matrix_regularized.transpose(-1, -2))

    return cov_matrix_regularized