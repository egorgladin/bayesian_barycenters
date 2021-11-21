import torch
import matplotlib.pyplot as plt
import pickle
import os.path

from algorithm import algorithm


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
    return (margin_r ** 2).sum(), (margin_c ** 2).sum()


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
    return r, Xs  # fixme reshape Xs from (m, n^2) to (m, n, n)

def print_cost_mat(device):
    for im_sz in [2, 3]:
        C = get_cost_matrix(im_sz, device)
        print(C)


def get_data_and_solution(device):
    ones = torch.ones((3, 1), device=device) / 3
    c1 = ones * torch.tensor([1., 0, 0], device=device)
    c2 = ones * torch.tensor([0., 0, 1], device=device)
    barycenter = ones * torch.tensor([0., 1, 0], device=device)
    cs = torch.stack([c1.flatten(), c2.flatten()])
    return cs, barycenter.flatten()


def get_optimal_plans(device):
    im_sz = 3
    X1 = torch.zeros(im_sz**2, im_sz**2, device=device)
    X2 = torch.zeros_like(X1)
    for i, idx_in_r in enumerate([1, 4, 7]):
        X1[idx_in_r, i*3] = 1. / 3.
        X2[idx_in_r, i*3 + 2] = 1. / 3.
    return torch.stack([X1, X2])


def print_data_and_solution(device):
    cs, r = get_data_and_solution(device)
    c1, c2 = cs[0], cs[1]
    for img in [c1, c2, r]:
        print(img.reshape(3, 3))


def check_plans(device, Xs):
    cs, r = get_data_and_solution(device)
    residue_r, residue_c = marginals_residuals(r, Xs, cs)
    print('Marginals residuals:', residue_r.item(), residue_c.item())


def compare_objective(device):
    cs, r = get_data_and_solution(device)
    Xs = get_optimal_plans(device)
    check_plans(device, Xs)

    X1_unoptimal = r.unsqueeze(1) @ cs[0].unsqueeze(0)
    X2_unoptimal = r.unsqueeze(1) @ cs[1].unsqueeze(0)
    Xs_unoptimal = torch.stack([X1_unoptimal, X2_unoptimal])
    check_plans(device, Xs_unoptimal)

    im_sz = 3
    cost_mat = get_cost_matrix(im_sz, device)
    kappas = [torch.tensor(1.), torch.tensor(1.)]
    value_optimal = barycenter_objective(r, Xs, cost_mat, kappas, cs)
    value_unoptimal = barycenter_objective(r, Xs_unoptimal, cost_mat, kappas, cs)
    print(f"Optimal value: {value_optimal}, unoptimal value: {value_unoptimal}")


def main():
    pass


if __name__ == "__main__":
    device = 'cuda'
    # print_cost_mat(device)
    # print_data_and_solution(device)
    # Xs = get_optimal_plans(device)
    # check_plans(device, Xs)
    compare_objective(device)
