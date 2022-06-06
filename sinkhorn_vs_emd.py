import torch
import numpy as np
import ot

from tqdm import tqdm

from core.essential_utils import get_cost_mat
from confidence_set_experiment import get_pairwise_dist, get_obj_vals_and_bary
from utils import replace_zeros


def get_pairwise_dist_lp(sample, cost_mat, dist_path):
    # sample has shape (n, n_samples)
    n_samples = sample.shape[1]
    sample = sample.cpu().type(torch.float64).numpy()
    sample /= sample.sum(axis=0)
    dist_mat = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples - 1)):
        wass_dist = ot.emd2(
            np.ascontiguousarray(sample[:, i]),
            np.ascontiguousarray(sample[:, i+1:]),
            cost_mat)
        dist_mat[i, i+1:] = wass_dist

    dist_mat = torch.tensor(dist_mat, dtype=torch.float32, device='cuda')
    dist_mat = dist_mat + dist_mat.T.contiguous()
    torch.save(dist_mat, dist_path[:-3] + '_lp.pt')
    return dist_mat


def main(n_samples, batch_sz, folder, prior_std, kappa,
         sinkhorn_reg=1e-2, replace_val=1e-6, max_samples=1000, dist_path=None):
    sampling_params = f"test_{n_samples}_{max_samples}_{prior_std}"
    if dist_path is None:
        dist_path = folder + sampling_params + "_dist.pt"

    device = 'cuda'
    mean_potentials = torch.load(folder + 'Mean.pt', map_location=device).detach()  # (mn,)

    img_sz = 28
    dtype = mean_potentials.dtype
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)
    _, bary_sample = get_obj_vals_and_bary(n_samples, prior_std, mean_potentials, folder, batch_sz,
                                           cost_mat, kappa, 1., max_samples, device=device)
    bary_sample = bary_sample.T.contiguous()
    bary_sample = replace_zeros(bary_sample, replace_val=replace_val, sumdim=0)

    cost_mat64 = get_cost_mat(img_sz, 'cpu', dtype=torch.float64).numpy()
    dist_mat_lp = get_pairwise_dist_lp(bary_sample, cost_mat64, dist_path)
    # dist_mat_lp = torch.load(dist_path[:-3] + '_lp.pt')

    dist_mat = get_pairwise_dist(bary_sample, cost_mat, dist_path, sinkhorn_reg=sinkhorn_reg)
    err_sum = torch.abs(dist_mat_lp - dist_mat).sum()
    print(f"Abs err sum {err_sum:.2f}")
    dist_sum = dist_mat_lp.sum()
    print(f"Rel err {100 * err_sum / dist_sum:.2f}%")


if __name__ == '__main__':
    n_samples = 20000
    max_samples = 40
    batch_sz = 40
    prior_std = 1e-1
    kappa = 1. / 30.
    sinkhorn_reg = 2e-2
    replace_val = 1e-7

    folder = ''
    barys_folder = 'for_colab'

    main(n_samples, batch_sz, folder, prior_std, kappa,
         sinkhorn_reg=sinkhorn_reg, max_samples=max_samples, replace_val=replace_val)
