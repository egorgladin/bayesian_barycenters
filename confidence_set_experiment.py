import torch
import ot
from math import ceil

from core.essential_utils import get_cost_mat
from core.compute_posterior import objective_function
from utils import replace_zeros


def sample_prior(n_samples, prior_std, mean_potentials, seed=42):
    prior_mean = mean_potentials.reshape(1, -1).expand(n_samples, -1)  # (n_samples, mn)
    torch.manual_seed(seed)
    return torch.normal(prior_mean, prior_std)  # (n_samples, mn)


def get_weights(potentials_sample, cost_mat, kappa, temperature, batch_sz=40, device='cuda'):
    # Can use different kappas for objective and mapping to simplex
    cs = torch.load('Archetypes.pt', map_location=device)

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)
    objective_values = []
    for i in range(ceil(potentials_sample.shape[0] / batch_sz)):
        batch = potentials_sample[i*batch_sz:(i+1)*batch_sz]
        objective_batch = objective(batch)
        objective_values.append(objective_batch)

    objective_values = torch.cat(objective_values)
    weights = torch.softmax(temperature * objective_values, dim=-1)
    return weights


def get_rho(x, sample, cost_mat, wght, sinkhorn_reg=1e-2, batch_sz=40):
    wass_dist = []
    for i in range(ceil(sample.shape[1] / batch_sz)):
        batch = sample[:, i*batch_sz:(i+1)*batch_sz]
        batch_dist = ot.sinkhorn2(x, batch, cost_mat, reg=sinkhorn_reg)
        wass_dist.append(batch_dist)

    wass_dist = torch.cat(wass_dist)
    rho = wght @ wass_dist
    return rho.item()


def main(n_samples, prior_std, kappa, temperature, device='cuda', sinkhorn_reg=1e-2, replace_val=1e-9):
    # Sample potentials from prior
    mean_potentials = torch.load('Mean.pt', map_location=device).detach()  # (mn,)
    print(mean_potentials.shape[0] / (28**2))
    potentials_sample = sample_prior(n_samples, prior_std, mean_potentials)  # (n_samples, mn)

    # Caclulate respective barycenters
    img_sz = 28
    potentials_sums = potentials_sample.reshape(n_samples, -1, img_sz**2).sum(dim=1)  # (n_samples, n)
    bary_sample = torch.softmax(-potentials_sums / kappa, dim=1)  # (n_samples, n)
    bary_sample = bary_sample.T.contiguous()  # (n, n_samples) - transposed for ot.sinkhorn2

    # Calculate respective weights
    dtype = potentials_sample.dtype
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)  # maybe minus in objective
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
    # Check if barycenters of other collections of 5s and 6s fall into CS
    print(f"Confidence set radius: {r:.2e}\nrho of prior_mean: {rho_prior_mean:.2e}")


if __name__ == '__main__':
    preparation = True

    if preparation:
        from utils import load_mnist, show_barycenter
        # RealBary = torch.load("RealBarycenter.pt")
        # fname = 'RealBarycenter'
        # show_barycenter(RealBary, fname, im_sz=28)  # 6

        m = 25
        device = 'cuda'
        im_sz = 28
        size = (im_sz, im_sz)
        cost_mat = get_cost_mat(im_sz, device)

        barys = []
        for target_digit in [5, 6]:
            for seed in [4, 5, 14, 23, 39]:
                print(seed)
                cs = load_mnist(m, target_digit, device, size=size, seed=seed)
                replace_val = 1e-6
                cs = replace_zeros(cs, replace_val=replace_val)
                reg = 0.002
                r = ot.barycenter(cs.T.contiguous(), cost_mat, reg, numItermax=20000)
                barys.append(r.clone())

            torch.save(torch.stack(barys), f'for_colab/barys{target_digit}.pt')
            barys = []
        # show_barycenter(r, f'barycenter{target_digit}_{seed}', im_sz=28, custom_folder='for_colab')

    else:
        n_samples = 80  # millions
        prior_std = 1e-3  #or -4
        kappa = 1. / 30.
        temperature = 1.
        sinkhorn_reg = 1e-2  # 1e-3
        # replace_val = 1e-6  # yes
        main(n_samples, prior_std, kappa, temperature, sinkhorn_reg=sinkhorn_reg)  #, replace_val=replace_val)
