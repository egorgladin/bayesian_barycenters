import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import ot

from algorithm import algorithm
from utils import load_data, safe_log, plot_trajectory, replace_zeros, get_sampler
from experiment_barycenter import get_cost_matrix


def experiment(n_steps, sample_size, prior_var, var_decay, noise_level=None, decay='exp', empir_cov=False,
               temperature=None, plot=False, track_time=False):
    dtype = torch.float32
    device = 'cuda'

    n_data_points = 9
    src_digit = 5
    target_digit = 3
    r_prior, cs = load_data(n_data_points, src_digit, target_digit, device, noise=noise_level)
    cs = replace_zeros(cs)

    img_size = 8
    cost_mat = get_cost_matrix(img_size, device, dtype=dtype)

    def objective(sample):
        # sample has shape (sample_size, n)
        barycenters = torch.softmax(sample, dim=-1).T.contiguous()  # shape (n, sample_size)
        wasser_dist = [ot.sinkhorn2(c, barycenters, cost_mat, reg=1e-2) for c in cs]

        # wasser_dist is list of (sample_size,) tensors
        objective_val = torch.vstack(wasser_dist).sum(dim=0)
        if temperature:
            objective_val *= temperature
        return -objective_val  # minus because algorithm maximizes objective

    z_prior = safe_log(r_prior)
    prior_cov = prior_var * torch.eye(img_size**2, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    def recalculate_cov(old_cov, sample, step, weights):
        factor = var_decay ** step if decay == 'exp' else var_decay / (step + var_decay)
        matrix = torch.cov(sample.T, aweights=weights) if empir_cov else prior_cov
        return factor * matrix

    trajectory = algorithm(z_prior, prior_cov, get_sample, n_steps, objective,
                           recalculate_cov, seed=0, track_time=track_time)

    if plot:
        # Names and indices of elements of the list returned by 'get_info'
        # info_names = [{'Objective': 1}]
        n_cols = 6
        img_name = f"IMG_samples_{sample_size}_var_{prior_var}_decay_{var_decay}"
        if empir_cov:
            img_name = 'cov_' + img_name

        plot_trajectory(trajectory, n_cols, img_size, img_name, None)

    else:
        return sum([info[2] for info in trajectory[-3:]]) / 3, trajectory


def main():
    n_steps = 100
    sample_size = 30000
    prior_var = 8.
    var_decay = 20.
    experiment(n_steps, sample_size, prior_var, var_decay, noise_level=0.1, decay='lin',
               plot=True, track_time=True)


if __name__ == "__main__":
    main()
