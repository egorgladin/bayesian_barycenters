"""
Experiment: find maximizer of a concave quadratic function.
"""
import torch
import math
from utils import get_cost_mat, safe_log


from experiment_barycenter import marginals_residuals,\
    barycenter_objective, get_data_and_solution, map_to_simplex, get_optimal_plans


def print_cost_mat(device):
    for im_sz in [2, 3]:
        C = get_cost_mat(im_sz, device)
        print(C)


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
    cost_mat = get_cost_mat(im_sz, device)
    kappas = [torch.tensor(1.), torch.tensor(1.)]
    value_optimal = barycenter_objective(r, Xs, cost_mat, kappas, cs)
    value_unoptimal = barycenter_objective(r, Xs_unoptimal, cost_mat, kappas, cs)
    print(f"Optimal value: {value_optimal}, unoptimal value: {value_unoptimal}")


def check_inverse(device):
    cs, r = get_data_and_solution(device)
    c1, c2 = cs[0], cs[1]
    Xs = get_optimal_plans(device)
    Ys = safe_log(Xs)
    n = 9

    for distr in [c1, c2, r]:
        z = safe_log(distr, minus_inf=-100.)
        sample = torch.cat((z, Ys.flatten()))
        r_, Xs_ = map_to_simplex(sample, n)
        isclose = torch.allclose(r_, distr, rtol=0., atol=math.exp(-100.), equal_nan=False)
        print(f"Is close? {isclose}")


if __name__ == "__main__":
    device = 'cuda'
    print_cost_mat(device)
    print_data_and_solution(device)
    Xs = get_optimal_plans(device)
    check_plans(device, Xs)
    compare_objective(device)
    check_inverse(device)
