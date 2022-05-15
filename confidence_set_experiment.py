import torch
import ot

from core.essential_utils import get_cost_mat
from core.compute_posterior import objective_function
from utils import replace_zeros


def sample_prior(n_samples, prior_std, mean_potentials, seed=42):
    prior_mean = mean_potentials.reshape(1, -1).expand(n_samples, -1)  # (n_samples, mn)
    torch.manual_seed(seed)
    return torch.normal(prior_mean, prior_std)  # (n_samples, mn)


def get_weights(potentials_sample, cost_mat, kappa, temperature, device='cuda'):
    # Can use different kappas for objective and mapping to simplex
    cs = torch.load('Archetypes.pt', map_location=device)

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)
    objective_values = objective(potentials_sample)  # (n_samples,)
    weights = torch.softmax(temperature * objective_values, dim=-1)
    return weights


def get_rho(x, sample, cost_mat, wght, sinkhorn_reg=1e-2, replace_val=1e-9):
    # x = replace_zeros(x, replace_val=replace_val)
    # sample = replace_zeros(sample, replace_val=replace_val, sumdim=0)
    wass_dist = ot.sinkhorn2(x, sample, cost_mat, reg=sinkhorn_reg)
    rho = wght @ wass_dist
    return rho.item()


def main(n_samples, prior_std, kappa, temperature, device='cuda', sinkhorn_reg=1e-2, replace_val=1e-9):
    # Sample potentials from posterior
    mean_potentials = torch.load('Mean.pt', map_location=device).detach()  # (mn,)
    potentials_sample = sample_prior(n_samples, prior_std, mean_potentials)  # (n_samples, m, n)

    # Caclulate respective barycenters
    img_sz = 28
    potentials_sums = potentials_sample.reshape(n_samples, -1, img_sz**2).sum(dim=1)  # (n_samples, n)
    bary_sample = torch.softmax(-potentials_sums / kappa, dim=1)  # (n_samples, n)
    bary_sample = bary_sample.T.contiguous()  # (n, n_samples) - transposed for ot.sinkhorn2

    # Calculate respective weights
    dtype = potentials_sample.dtype
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)
    weights = get_weights(potentials_sample, cost_mat, kappa, temperature, device=device)

    # Calculate rho for all barycenters
    rhos = []
    for i in range(n_samples):
        src = bary_sample[:, i]
        dst = torch.hstack((bary_sample[:, :i],
                            bary_sample[:, i+1:]))
        wght = torch.cat((weights[:i], weights[i+1:]))
        rho = get_rho(src, dst, cost_mat, wght, sinkhorn_reg=sinkhorn_reg, replace_val=replace_val)
        rhos.append(rho)

    # Compute radius of conf set
    pairs = list(zip(rhos, weights.tolist()))
    sorted_pairs = sorted(pairs, key=lambda tup: tup[0])
    threshold = 0.95
    weight_in_CS = 0.
    i = 0
    while (weight_in_CS < threshold) and (i < n_samples):
        weight_in_CS += sorted_pairs[i][1]
        i += 1

    r = sorted_pairs[i][0]

    # Check if prior mean is in the CS
    mean_poten_sum = mean_potentials.reshape(-1, img_sz**2).sum(dim=0)  # (n,)
    prior_mean_bary = torch.softmax(-mean_poten_sum / kappa, dim=0)  # (n,)
    rho_prior_mean = get_rho(prior_mean_bary, bary_sample, cost_mat, weights,
                             sinkhorn_reg=sinkhorn_reg, replace_val=replace_val)

    # Check if "true" bary is in the CS
    # Check if other images fall into confidence set
    print(f"Confidence set radius: {r:.2e}\nrho of prior_mean: {rho_prior_mean:.2e}")


if __name__ == '__main__':
    n_samples = 50
    prior_std = 10.
    kappa = 1. / 30.
    temperature = 1.
    sinkhorn_reg = 1e-2
    # replace_val = 1e-6
    main(n_samples, prior_std, kappa, temperature, sinkhorn_reg=sinkhorn_reg)  #, replace_val=replace_val)
