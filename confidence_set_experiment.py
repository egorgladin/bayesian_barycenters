import torch
import ot
from math import ceil
import numpy as np

from core.essential_utils import get_cost_mat
from core.compute_posterior import objective_function
from utils import replace_zeros, get_sample_generator

from tqdm import tqdm


def get_weights_and_bary(n_samples, prior_std, mean_potentials, data_path, batch_sz, cost_mat, kappa, temperature,
                         device='cuda'):
    prior_mean = mean_potentials.reshape(1, -1).expand(batch_sz, -1)
    n_batches = ceil(n_samples / batch_sz)
    n = cost_mat.shape[0]
    sample_generator = get_sample_generator(prior_mean, n_batches, prior_std)

    cs = torch.load(data_path, map_location=device)
    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)

    objective_values = []
    barys = []
    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        # objective
        objective_batch = objective(batch)
        objective_values.append(objective_batch)

        # bary
        potentials_sums = batch.reshape(batch_sz, -1, n).sum(dim=1)  # (n_samples, n)
        bary_batch = torch.softmax(-potentials_sums / kappa, dim=1)
        barys.append(bary_batch)

    # weights
    objective_values = torch.cat(objective_values)
    weights = torch.softmax(temperature * objective_values, dim=-1)
    return weights, torch.vstack(barys)


def sample_prior(n_samples, prior_std, mean_potentials, seed=42):
    prior_mean = mean_potentials.reshape(1, -1).expand(n_samples, -1)  # (n_samples, mn)
    torch.manual_seed(seed)
    return torch.normal(prior_mean, prior_std)  # (n_samples, mn)


def get_weights(potentials_sample, data_path, batch_sz, cost_mat, kappa, temperature, device='cuda'):
    # Can use different kappas for objective and mapping to simplex
    cs = torch.load(data_path, map_location=device)

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)
    objective_values = []
    n_batches = ceil(n_samples / batch_sz)
    for i in tqdm(range(n_batches)):
        batch = potentials_sample[i * batch_sz:(i + 1) * batch_sz]
        objective_batch = objective(batch)
        objective_values.append(objective_batch)

    objective_values = torch.cat(objective_values)
    weights = torch.softmax(temperature * objective_values, dim=-1)
    return weights


def get_rho(x, sample, cost_mat, wght, sinkhorn_reg=1e-2, batch_sz=40):
    wass_dist = ot.sinkhorn2(x, sample, cost_mat, reg=sinkhorn_reg)
    rho = wght @ wass_dist
    return rho.item()


def main(n_samples, batch_sz, mean_path, data_path, prior_std, kappa, temperature, device='cuda', barys_folder=None,
         sinkhorn_reg=1e-2, replace_val=1e-6, max_samples=1000):
    log_name = f"{n_samples}_{max_samples}_{prior_std}_{sinkhorn_reg}.txt"
    # Sample potentials from prior
    mean_potentials = torch.load(mean_path, map_location=device).detach()  # (mn,)

    img_sz = 28
    dtype = mean_potentials.dtype
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)

    old = False
    if old:
        potentials_sample = sample_prior(n_samples, prior_std, mean_potentials)  # (n_samples, mn)

        # Calculate respective weights
        # maybe minus in objective
        weights = get_weights(potentials_sample, data_path, batch_sz, cost_mat, kappa, temperature, device=device)

        # Caclulate respective barycenters
        potentials_sums = potentials_sample.reshape(n_samples, -1, img_sz ** 2).sum(dim=1)  # (n_samples, n)
        bary_sample = torch.softmax(-potentials_sums / kappa, dim=1)  # (n_samples, n)
        bary_sample = bary_sample.T.contiguous()  # (n, n_samples) - transposed for ot.sinkhorn2

    else:
        weights, bary_sample = get_weights_and_bary(n_samples, prior_std, mean_potentials, data_path, batch_sz,
                                                    cost_mat, kappa, temperature, device=device)

    # Preserve only top 'max_samples' elements
    if n_samples > max_samples:
        n_samples = max_samples
        top_idx = np.argsort(weights.cpu())[-max_samples:]
        # potentials_sample = potentials_sample[top_idx]
        bary_sample = bary_sample[top_idx]
        weights = weights[top_idx]

    bary_sample = bary_sample.T.contiguous()

    # Calculate rho for all barycenters
    rhos = []
    for i in tqdm(range(n_samples)):
        src = bary_sample[:, i]
        dst = torch.hstack((bary_sample[:, :i],
                            bary_sample[:, i + 1:]))
        wght = torch.cat((weights[:i], weights[i + 1:]))
        rho = get_rho(src, dst, cost_mat, wght, sinkhorn_reg=sinkhorn_reg)
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

    idx = i if i < n_samples else n_samples - 1
    r = sorted_pairs[idx][0]

    # Check if prior mean is in the CS
    mean_poten_sum = mean_potentials.reshape(-1, img_sz ** 2).sum(dim=0)  # (n,)
    prior_mean_bary = torch.softmax(-mean_poten_sum / kappa, dim=0)  # (n,)
    rho_prior_mean = get_rho(prior_mean_bary, bary_sample, cost_mat, weights,
                             sinkhorn_reg=sinkhorn_reg)

    log_path = '/content/drive/MyDrive/logs/'
    with open(log_path + log_name, "w") as handle:
        # Check if barycenters of other collections of 5s and 6s fall into CS
        if barys_folder:
            for target_digit in [5, 6]:
                handle.write(f"\nChecking barycenters of {target_digit}\n")
                barys = torch.load(barys_folder + f"/barys{target_digit}.pt", map_location=device)
                barys = replace_zeros(barys, replace_val=1e-6, sumdim=-1)
                for i in range(barys.shape[0]):
                    handle.write(f'{get_rho(barys[i], bary_sample, cost_mat, weights, sinkhorn_reg=sinkhorn_reg):.2e} ')

        # Check if "true" bary is in the CS
        handle.write(f"\nConfidence set radius: {r:.2e}\nrho of prior_mean: {rho_prior_mean:.2e}")


def prepare_for_experiment():
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


if __name__ == '__main__':
    preparation = True

    if preparation:
        prepare_for_experiment()

    else:
        n_samples = 1000000  # millions
        max_samples = 1000
        batch_sz = 100
        prior_std = 1e-2  # or -4
        kappa = 1. / 30.
        temperature = 1.
        sinkhorn_reg = 1e-2  # 1e-3
        # replace_val = 1e-6  # yes

        mean_path = '/content/drive/MyDrive/Mean.pt'
        data_path = '/content/drive/MyDrive/Archetypes.pt'
        barys_folder = '/content/drive/MyDrive/barycenters_for_test'

        main(n_samples, batch_sz, mean_path, data_path, prior_std, kappa, temperature, barys_folder=barys_folder,
             sinkhorn_reg=sinkhorn_reg, max_samples=max_samples)  # , replace_val=replace_val)

        log_path = '/content/drive/MyDrive/logs/'
        log_name = f"{n_samples}_{max_samples}_{prior_std}_{sinkhorn_reg}.txt"
        with open(log_path + log_name, 'r') as handle:
            print(handle.read())
