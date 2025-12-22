import torch
from typing import Optional


def log_marginal_likelihood(
    weights: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the log-marginal-likelihood of the data given the weights.

    Parameters
    ----------
    weights: torch.Tensor
        weights of the clusters.
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from cluster j.

    Returns
    -------
    log_marginal_likelihood: torch.Tensor

    """
    log_marginal_likelihood = torch.mean(torch.logsumexp(log_likelihood + torch.log(weights), axis=1))
    return log_marginal_likelihood


def grad_log_prob(
    weights: torch.Tensor,
    log_likelihood: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate the gradient of the log-likelihood of the data given the weights.
    This is computed in log-space.

    If not in log-space, this would be something like:    

        # make each row of loglikelihood have max at 0 <-> likelihood row max at 1
        # (grad is invariant to scaling each likelihood row by a constant)
        log_likelihood = log_likelihood - torch.max(log_likelihood, 1)[:, None]
        likelihood = np.exp(log_likelihood)
        
        density_at_weights = torch.sum(likelihood*weights, axis=1)
        grad = torch.mean(likelihood/density_at_weights[:, None], axis=0)

    
    Parameters
    ----------
    weights: torch.Tensor
        weights of the clusters.
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from cluster j.

    Returns
    -------
    grad: torch.Tensor

    """
    num_images, num_structures = log_likelihood.shape

    log_density_at_weights = torch.logsumexp(log_likelihood + torch.log(weights), axis=1)

    aux = log_likelihood - log_density_at_weights.reshape(num_images, 1)
    grad =  (1/num_images)*(torch.exp(torch.logsumexp(aux, axis=0)))
    return grad


def multiplicative_gradient(
    log_likelihood,
    tol: Optional[float]=10**-3,
    max_iterations: Optional[int]=10000,
    stats_frequency: Optional[int]=100,
    VERBOSE: Optional[bool]=False,
    TRACK_WEIGHTS: Optional[bool] = False
)->float:
    
    """
     This function updates the weights according to the expectation maximization
     algorithm for mixture models.
     This is also known as the "multiplicative gradient" method, which has much less notation overload with "EM".
     
     By default, the initial weights are set to equal probabilities for all structures, the `most entropic' weights.
 
    Parameters
    ----------
    log_likelihood: torch.Tensor
        Log-likelihood of generating image i from structure j.
    tol: float
        Tolerance for the stopping criteria
    max_iterations: int
        Max iterations if stopping criteria isn't met
    stats_frequency: int
        Stats are computed at every (stats frequency) iterations
    VERBOSE: bool
        If TRUE, some print statements will happen every stats_frequency iterations
    TRACK_WEIGHTS: bool
        If TRUE, the weights at every stats_frequency iterations will be saved

    Returns
    -------
    weights: torch.tensor 
    stats: dictionary
    """
    num_images, num_structures = log_likelihood.shape

    # Initialize Weights
    weights = (1/num_structures)*torch.ones(num_structures)

    stats = {}
    stats["losses"] = []
    stats["idx"] = []
    stats["weights"] = []
    #stats_tracking["your_favorite_stat"] = []

    # Iterate
    for k in range(max_iterations):
        
        # Update stats 
        if k % stats_frequency == 0: 
            loss = -1*log_marginal_likelihood(weights, log_likelihood)
            stats["losses"].append(loss)
            stats["idx"].append(k)
            #stats_tracking["your_favorite_stat"].append(...)

            if TRACK_WEIGHTS:
                stats["weights"].append(weights)
            
            if VERBOSE:
                print(f"#iterations: {k}")
                print(f"loss: {loss}")
                print("\n")
        
        # Update weights
        grad = grad_log_prob(weights, log_likelihood)   
        weights = weights*grad

        # Check stopping criterion: this `gap` is an upper bound on our loss compared to max likelihood weights
        gap = torch.max(grad) - 1

        if gap < tol:
            print("exiting!")
            print(f"#iterations at exit: {k}")
            break

    # Collect stats
    stats["losses"] = torch.stack(stats["losses"])
    stats["idx"] = torch.tensor(stats["idx"])
    if TRACK_WEIGHTS:
        stats["weights"] = torch.stack(stats["weights"])
    
    return weights, stats