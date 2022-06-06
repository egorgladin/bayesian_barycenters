import torch
import ot
from math import ceil
import numpy as np

from core.essential_utils import get_cost_mat
from core.compute_posterior import objective_function
from utils import replace_zeros, get_sample_generator

from tqdm import tqdm


def get_obj_vals_and_bary(n_samples, prior_std, mean_potentials, folder, batch_sz, cost_mat,
                          kappa, temperature, max_samples, device='cuda'):
    prior_mean = mean_potentials.reshape(1, -1).expand(batch_sz, -1)
    n_batches = ceil(n_samples / batch_sz)
    n = cost_mat.shape[0]
    sample_generator = get_sample_generator(prior_mean, n_batches, prior_std)

    cs = torch.load(folder + 'Archetypes.pt', map_location=device)
    m = cs.shape[0]
    objective = lambda sample: objective_function(temperature * sample, cost_mat, cs, kappa)

    best = torch.empty(max_samples + batch_sz, n, device=device)
    obj_vals = torch.empty(max_samples + batch_sz, device=device)

    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        position = i * batch_sz
        if position > (max_samples - batch_sz):
            best[max_samples:] = torch.softmax(-batch.reshape(batch_sz, -1, n).sum(dim=1) / kappa, dim=1)
            obj_vals[max_samples:] = objective(batch)
            obj_vals, indices = torch.sort(obj_vals)
            best = best[indices]
        else:
            best[position:position + batch_sz] = torch.softmax(-batch.reshape(batch_sz, -1, n).sum(dim=1) / kappa, dim=1)
            obj_vals[position:position + batch_sz] = objective(batch)

    return obj_vals[:max_samples], best[:max_samples]


def get_pairwise_dist(sample, cost_mat, dist_path, sinkhorn_reg=1e-2):
    # sample has shape (n, n_samples)
    n_samples = sample.shape[1]
    dist_mat = torch.zeros(n_samples, n_samples, device=sample.device)
    for i in tqdm(range(n_samples - 1)):
        src = sample[:, i]
        dst = sample[:, i + 1:]
        wass_dist = ot.sinkhorn2(src, dst, cost_mat, reg=sinkhorn_reg)
        dist_mat[i, i + 1:] = wass_dist

    dist_mat = dist_mat + dist_mat.T.contiguous()
    torch.save(dist_mat, dist_path)
    return dist_mat


def get_pairwise_dist_lp(sample, cost_mat, dist_path):
    # sample has shape (n, n_samples)
    n_samples = sample.shape[1]
    sample = sample.cpu().type(torch.float64).numpy()
    sample /= sample.sum(axis=0)
    dist_mat = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples - 1)):
        src = np.ascontiguousarray(sample[:, i])
        dst = np.ascontiguousarray(sample[:, i + 1:])
        wass_dist = ot.emd2(src, dst, cost_mat)
        dist_mat[i, i + 1:] = wass_dist

    dist_mat = torch.tensor(dist_mat, dtype=torch.float32, device='cuda')
    dist_mat = dist_mat + dist_mat.T.contiguous()
    # dist_mat = dist_mat.type(torch.float64).to(sample.device)
    torch.save(dist_mat, dist_path[:-3] + '_lp.pt')
    return dist_mat


def get_rho(x, sample, cost_mat, wght, sinkhorn_reg=1e-2, batch_sz=40):
    wass_dist = ot.sinkhorn2(x, sample, cost_mat, reg=sinkhorn_reg)
    rho = wght @ wass_dist
    return rho.item()


def main(n_samples, batch_sz, folder, prior_std, kappa, temperature, device='cuda', barys_folder=None,
         sinkhorn_reg=1e-2, replace_val=1e-6, max_samples=1000, load_distances=False, dist_path=None):
    sampling_params = f"{n_samples}_{max_samples}_{prior_std}"
    log_name = sampling_params + f"_{sinkhorn_reg}.txt"
    if dist_path is None:
        dist_path = folder + sampling_params + "_dist.pt"

    mean_potentials = torch.load(folder + 'Mean.pt', map_location=device).detach()  # (mn,)

    img_sz = 28
    dtype = mean_potentials.dtype
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)

    objective_vals, bary_sample = get_obj_vals_and_bary(n_samples, prior_std, mean_potentials, folder,
                                                        batch_sz, cost_mat, kappa, temperature, max_samples,
                                                        device=device)
    bary_sample = replace_zeros(bary_sample, replace_val=replace_val, sumdim=-1)
    bary_sample = bary_sample.T.contiguous()

    # Calculate rho for all barycenters
    dist_mat = torch.load(dist_path) if load_distances \
        else get_pairwise_dist(bary_sample, cost_mat, dist_path, sinkhorn_reg=sinkhorn_reg)

    cost_mat64 = get_cost_mat(img_sz, 'cpu', dtype=torch.float64).numpy()
    dist_mat_lp = get_pairwise_dist_lp(bary_sample, cost_mat64, dist_path)
    dist_mat_lp = torch.load(dist_path[:-3] + '_lp.pt')
    err_sum = torch.abs(dist_mat_lp - dist_mat).sum()
    dist_sum = dist_mat_lp.sum()
    print(f"Rel err {100 * err_sum / dist_sum:.2f}%")

    # err_sum1 = torch.abs(dist_mat_lp[0, 1:] - dist_mat[0, 1:]).sum()
    # dist_sum1 = dist_mat_lp[0, 1:].sum()
    # print(f"Rel err1 {100 * err_sum1 / dist_sum1:.2f}%")

    rhos = dist_mat @ weights

    # Compute radius of conf set
    pairs = list(zip(rhos.tolist(), weights.tolist()))
    sorted_pairs = sorted(pairs, key=lambda tup: tup[0])
    threshold = 0.95
    weight_in_CS = 0.
    i = 0
    while (weight_in_CS < threshold) and (i < max_samples):
        weight_in_CS += sorted_pairs[i][1]
        i += 1

    idx = i if i < max_samples else max_samples - 1
    r = sorted_pairs[idx][0]

    # Check if prior mean is in the CS
    mean_poten_sum = mean_potentials.reshape(-1, img_sz ** 2).sum(dim=0)  # (n,)
    prior_mean_bary = torch.softmax(-mean_poten_sum / kappa, dim=0)  # (n,)
    prior_mean_bary = replace_zeros(prior_mean_bary, replace_val=replace_val, sumdim=-1)
    rho_prior_mean = get_rho(prior_mean_bary, bary_sample, cost_mat, weights,
                             sinkhorn_reg=sinkhorn_reg)

    with open(folder + 'logs/' + log_name, "w") as handle:
        # Check if barycenters of other collections of 5s and 6s fall into CS
        if barys_folder:
            for target_digit in [5, 6]:
                handle.write(f"\nChecking barycenters of {target_digit}\n")
                barys = torch.load(barys_folder + f"/barys{target_digit}.pt", map_location=device)
                barys = replace_zeros(barys, replace_val=replace_val, sumdim=-1)
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
        n_samples = 5000000  # millions
        max_samples = 2000
        batch_sz = 100
        prior_std = 1e-1
        kappa = 1. / 30.
        temperature = 2.
        sinkhorn_reg = 2e-2
        replace_val = 1e-9

        folder = '/content/drive/MyDrive/'
        barys_folder = '/content/drive/MyDrive/barycenters_for_test'

        main(n_samples, batch_sz, folder, prior_std, kappa, temperature, barys_folder=barys_folder,
             sinkhorn_reg=sinkhorn_reg, max_samples=max_samples, replace_val=replace_val)

        log_path = '/content/drive/MyDrive/logs/'
        log_name = f"{n_samples}_{max_samples}_{prior_std}_{sinkhorn_reg}.txt"
        with open(log_path + log_name, 'r') as handle:
            print(handle.read())
