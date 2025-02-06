import torch
from torch.distributions import MixtureSameFamily, Categorical, MultivariateNormal


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


def batched_gmm_sample_with_clusters(gmm, num_samples, batch_size=100):
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

    for _ in range(0, num_samples, batch_size):
        batch_size_actual = min(
            batch_size, num_samples - len(samples)
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

    return torch.cat(samples, dim=0), torch.cat(cluster_indices, dim=0)


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

def load_gmm(weights_path, means_path, covariance_path):
    """
    Load a Gaussian Mixture Model (GMM) from saved files.

    Args:
        weights_path (str): Path to the weights file
        means_path (str): Path to the means file
        covariance_path (str): Path to the covariance file

    Returns:
        MixtureSameFamily: A PyTorch GMM distribution object
    """
    weights = torch.load(weights_path, weights_only=True)
    means = torch.load(means_path, weights_only=True)
    covariances = torch.load(covariance_path, weights_only=True)

    return create_gmm(weights, means, covariances)