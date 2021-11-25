import torch
import matplotlib.pyplot as plt
import pickle
import os.path

from algorithm import algorithm
from utils import safe_log, plot_trajectory, norm_sq


def get_cost_matrix(im_sz, device):
    """
    Compute ground cost matrix for images.

    :param im_sz: positive int
    :param device: 'cpu' or 'cuda'
    :return: (im_sz^2, im_sz^2) tensor
    """
    C = torch.zeros(im_sz**2, im_sz**2, device=device)
    for i in range(im_sz):
        for j in range(im_sz):
            I = im_sz*i + j
            for k in range(im_sz):
                for l in range(im_sz):
                    J = im_sz*k + l
                    C[I, J] = (i - k)**2 + (j - l)**2
    return C / (im_sz - 1)**2


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


def map_to_simplex(theta, n):
    """
    Map a vector of parameters from R^d to points on unit simplexes.

    :param theta: (d,) tensor with d = n + m*n^2
    :param n: dimension of barycenter
    :return: list of (n,) tensor (barycenter) and (m, n, n) tensor (plans)
    """
    z = theta[:n]
    Ys = theta[n:].reshape(-1, n**2)
    r = torch.softmax(z, dim=-1)
    Xs = torch.softmax(Ys, dim=-1)
    return r, Xs.reshape(-1, n, n)


def get_data_and_solution(device):
    """
    Get two simple 3x3 images and their barycenter.

    :param device: 'cuda' or 'cpu'
    :return: list of (2, 9) tensor (data points) and (9,) tensor (barycenter)
    """
    ones = torch.ones((3, 1), device=device) / 3
    c1 = ones * torch.tensor([1., 0, 0], device=device)
    c2 = ones * torch.tensor([0., 0, 1], device=device)
    barycenter = ones * torch.tensor([0., 1, 0], device=device)
    cs = torch.stack([c1.flatten(), c2.flatten()])
    return cs, barycenter.flatten()


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


def main(kappa, std_decay, sample_size, n_steps, prior_std, noise_level):
    device = 'cuda'
    im_sz = 3
    n = im_sz**2
    cost_mat = get_cost_matrix(im_sz, device)
    cs, r_opt = get_data_and_solution(device)
    Xs_opt = get_optimal_plans(device)
    kappas = [kappa, kappa]

    def objective(sample):
        sample_size = sample.shape[0]
        values = torch.empty(sample_size, device=device)
        for i in range(sample_size):
            r, Xs = map_to_simplex(sample[i], n)
            values[i] = barycenter_objective(r, Xs, cost_mat, kappas, cs)
        return values

    noise = torch.normal(torch.zeros_like(r_opt), noise_level)
    r_prior = r_opt + torch.where(noise > 0., noise, torch.tensor(0., device=device))
    r_prior /= r_prior.sum()

    X1 = r_prior.unsqueeze(1) @ cs[0].unsqueeze(0)
    X2 = r_prior.unsqueeze(1) @ cs[1].unsqueeze(0)
    Xs = torch.stack([X1, X2])

    z_prior = safe_log(r_prior, device)
    Ys = safe_log(Xs, device)
    prior_mean = torch.cat((z_prior, Ys.flatten()))

    def get_info(theta):  # store only barycenters in trajectory
        r, Xs = map_to_simplex(theta, n)
        residue_r, residue_c = marginals_residuals(r, Xs, cs)
        acc_X1 = norm_sq(Xs[0] - Xs_opt[0])
        acc_X2 = norm_sq(Xs[1] - Xs_opt[1])
        acc_r = norm_sq(r - r_opt)
        return r, residue_r, residue_c, acc_X1, acc_X2, acc_r  # theta[:n].clone()

    trajectory = algorithm(prior_mean, prior_std, n_steps, sample_size, objective,
                           std_decay=std_decay, seed=0, get_info=get_info, track_time=True)
    n_cols = 6
    img_name = f"kappa_{kappa}_sample_size_{sample_size}_prior_std_{prior_std}"
    plot_trajectory(trajectory, n_cols, im_sz, img_name)


if __name__ == "__main__":
    std_decay = 0.99
    noise_level = 0.05
    n_steps = 13
    sample_size = 1000
    kappa = 2.
    prior_std = 10.
    main(kappa, std_decay, sample_size, n_steps, prior_std, noise_level)
