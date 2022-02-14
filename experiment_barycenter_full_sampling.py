"""
Experiment: find barycenter for a simple 3x3 setup.
"""
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.stats

from algorithm import algorithm
from utils import safe_log, plot_trajectory, norm_sq
from experiment_barycenter import get_cost_matrix, get_data_and_solution, get_init_barycenter


def marginals_residuals(r, Xs, cs):
    """
    Compute squared norms of residuals for marginal distributions.

    :param r: (n,) tensor (barycenter)
    :param Xs: (m, n, n) tensor (transportation plans)
    :param cs: (m, n) tensor (distributions, data points)
    :return: list of 2 scalar tensors
    """
    margin_r = Xs.sum(dim=-1) - r  # (m, n) tensor
    margin_c = Xs.sum(dim=1) - cs  # (m, n) tensor
    return norm_sq(margin_r), norm_sq(margin_c)


def barycenter_objective(r, Xs, cost_mat, kappas, cs):
    """
    Compute objective value for barycenter problem.

    :param r: (n,) tensor (barycenter)
    :param Xs: (m, n, n) tensor (transportation plans)
    :param cost_mat: (n, n) tensor
    :param kappas: list of 2 scalar tensors (kappa_r and kappa_c)
    :param cs: (m, n) tensor (distributions, data points)
    :return: scalar tensor
    """
    transport_cost = torch.tensordot(Xs, cost_mat.unsqueeze(0), dims=3)  # scalar tensor
    penalty_r, penalty_c = marginals_residuals(r, Xs, cs)  # 2 scalar tensors
    kappa_r, kappa_c = kappas
    return transport_cost + kappa_r * penalty_r + kappa_c * penalty_c


def map_to_simplex(theta, n_):
    """
    Map a vector of parameters from R^d to points on unit simplexes.

    :param theta: (d,) tensor with d = n_ + m*n^2
    :param n_: dimension of barycenter
    :return: list of (n_,) tensor (barycenter) and (m, n_, n_) tensor (plans)
    """
    z = theta[:n_]
    Ys = theta[n_:].reshape(-1, n_ ** 2)
    r = torch.softmax(z, dim=-1)
    Xs = torch.softmax(Ys, dim=-1)
    return r, Xs.reshape(-1, n_, n_)


def get_optimal_plans(device):
    """
    Get optimal plans for simple 3x3 images.

    :param device: 'cuda' or 'cpu'
    :return: (2, 9, 9) tensor
    """
    im_sz = 3
    X1 = torch.zeros(im_sz**2, im_sz**2, device=device)
    X2 = torch.zeros_like(X1)
    for i, idx_in_r in enumerate([1, 4, 7]):
        X1[idx_in_r, i*3] = 1. / 3.
        X2[idx_in_r, i*3 + 2] = 1. / 3.
    return torch.stack([X1, X2])


def get_init_plans(r_prior, cs):
    """Define initial transport plans that give correct marginals."""
    X1 = r_prior.unsqueeze(1) @ cs[0].unsqueeze(0)
    X2 = r_prior.unsqueeze(1) @ cs[1].unsqueeze(0)
    Xs = torch.stack([X1, X2])

    # map to the whole Euclidean space
    Ys = safe_log(Xs)
    return Xs, Ys


def alg_full_sampling(img_size, kappa, var_decay, sample_size, n_steps, prior_std, device, noise_level=None):
    n = img_size ** 2  # barycenter dimensionality
    cost_mat = get_cost_matrix(img_size, device)
    cs, r_opt = get_data_and_solution(device)
    Xs_opt = get_optimal_plans(device)
    # kappas = [kappa, kappa]  # equal coefficients for penalizing both marginals
    kappas = [torch.tensor(kappa, device=device)] * 2

    def objective(sample):
        sample_size = sample.shape[0]
        values = torch.empty(sample_size, device=device)
        for i in range(sample_size):
            r, Xs = map_to_simplex(sample[i], n)
            values[i] = barycenter_objective(r, Xs, cost_mat, kappas, cs)
        return -values  # minus because the algorithm maximizes the objective

    # Define initial approx. barycenter and transport plans as well as their counterparts in R^n
    if noise_level is not None:
        r_prior, z_prior = get_init_barycenter(r_opt, noise_level, seed=0)  # initial approx. barycenter
    else:
        r_prior = torch.ones_like(r_opt)
        r_prior /= r_prior.sum()
        z_prior = safe_log(r_prior)
    Xs, Ys = get_init_plans(r_prior, cs)
    prior_mean = torch.cat((z_prior, Ys.flatten()))  # concatenate variables in one tensor

    # the following function is needed to store info about convergence of the algorithm
    def get_info(theta):
        """
        Get information about how far the current point from optimum.

        :param theta: vector of variables at the current step
        :return: list containing current approximation of barycenter and
            information about how far the current point from optimum
        """
        r, Xs = map_to_simplex(theta, n)
        residue_r, residue_c = marginals_residuals(r, Xs, cs)
        acc_X1 = norm_sq(Xs[0] - Xs_opt[0])  # distance between current and optimal plans
        acc_X2 = norm_sq(Xs[1] - Xs_opt[1])
        acc_r = norm_sq(r - r_opt)  # distance between current approximation and true barycenter
        print(f"Error of r: {acc_r:.3f}")
        objective_val = -objective(theta.unsqueeze(0)).item()
        transport_cost = torch.tensordot(Xs, cost_mat.unsqueeze(0), dims=3).item()
        return r, residue_r, residue_c, objective_val, transport_cost, acc_X1, acc_X2, acc_r

    # Names and indices of elements of the list returned by 'get_info'
    info_names = [{'Marginal error for r': 1, 'Marginal error for c': 2, 'Objective': 3, 'Transport cost': 4},
                  {'Error of X1': 5, 'Error of X2': 6}, {'Error of r': 7}]

    def get_sample(mean, prior_std, seed):
        if seed is not None:
            torch.manual_seed(seed)
        sample = torch.normal(mean.expand(sample_size, -1), prior_std)  # (sample_size, d)
        return sample

    def recalculate_cov(old_std, sample, step, weights):
        return var_decay * old_std

    trajectory = algorithm(prior_mean, prior_std, get_sample, n_steps, objective,
                           recalculate_cov, seed=0, get_info=get_info, track_time=True)
    n_cols = 6
    img_name = f"kappa_{kappa}_sample_size_{sample_size}_prior_std_{prior_std}"
    opt_val = barycenter_objective(r_opt, Xs_opt, cost_mat, kappas, cs)
    plot_trajectory(trajectory, n_cols, img_size, img_name, info_names, opt_val=opt_val)


def main():
    img_size = 3  # width = height of an image
    n_steps = 200  # number of iterations

    # Hyperparameters
    device = 'cpu'
    sample_size = 100000
    prior_std = 3.  # initial variance of prior
    std_decay = 0.98  # decay rate for variance
    kappa = 1.

    alg_full_sampling(img_size, kappa, std_decay, sample_size, n_steps, prior_std, device)

    # prior_stds = [2.5]  # test various standard deviations for sampling
    # std_decays = [0.9]  # test various decay rates of standard deviation for sampling
    # for std_decay in std_decays:
    #     for prior_std in prior_stds:
    #          experiment_pot(std_decay, sample_size, n_steps, float(prior_std), noise_level, device, add_entropy=False)


if __name__ == "__main__":
    main()
