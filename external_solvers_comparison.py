import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet
import ot
from experiment_barycenter import get_cost_matrix

import time
import csv


def gen_data(distr, size, seed1, seed2):
    torch.manual_seed(seed1)
    a = distr.sample()
    torch.manual_seed(seed2)
    b = distr.sample((size,))
    return a, b.T.contiguous()


def sinkhorn_comparison():
    n_experim = 2

    im_sz = 5
    n_samples = 10
    cost_mat = get_cost_matrix(im_sz, device='cuda', dtype=torch.float32)

    methods = ['sinkhorn', 'sinkhorn_stabilized', 'sinkhorn_log']  #
    reg_coeffs = [1e-1, 1e-2, 1e-3]  #
    results = {method: {reg_coeff: {'time': 0., 'relative_eror': 0.} for reg_coeff in reg_coeffs}
               for method in methods}
    exact_emds = []

    alpha = torch.ones(im_sz**2, device='cuda', dtype=torch.float32)
    distr = Dirichlet(alpha)

    # exact emd
    start = time.time()
    for i in range(n_experim):
        a, b = gen_data(distr, n_samples, i, n_experim + i)
        a = a.type(torch.float64)
        a /= a.sum()
        b = b.type(torch.float64)
        b /= b.sum(axis=0, keepdims=True)
        emds = ot.emd2(a, b, cost_mat)
        exact_emds.append(sum(emds))

    time_emd = (time.time() - start) / n_experim
    print(f'Time emd: {time_emd:.3f}s')  # Time emd: 19.778s

    # sinkhorn
    for reg_coeff in reg_coeffs:
        for alg in methods:
            start = time.time()
            for i in range(n_experim):
                a, b = gen_data(distr, n_samples, i, n_experim + i)
                emds = ot.sinkhorn2(a, b, cost_mat, method=alg, reg=reg_coeff)
                error = (sum(emds) - exact_emds[i]) / exact_emds[i]
                results[alg][reg_coeff]['relative_eror'] += abs(error.item()) / n_experim

            results[alg][reg_coeff]['time'] = (time.time() - start) / n_experim
            print(f"Time (reg {reg_coeff}, alg {alg}): {results[alg][reg_coeff]['time']:.3f}s")

    for metrics in ['time', 'relative_eror']:
        with open(f'solvers_comparison_{im_sz}x{im_sz}_{metrics}.csv', 'w', newline='') as csvfile:
            fieldnames = ['reg_coeff'] + methods
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for reg_coeff in reg_coeffs:
                row = {alg: results[alg][reg_coeff][metrics] for alg in methods}
                row['reg_coeff'] = reg_coeff
                writer.writerow(row)


if __name__ == "__main__":
    sinkhorn_comparison()

