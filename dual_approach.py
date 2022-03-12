import torch
import time
import gc

import numpy as np

from algorithm import algorithm
from experiment_barycenter import get_cost_matrix, get_data_and_solution
from utils import plot_trajectory, norm_sq, get_sampler, plot3d


def get_factor(decay, var_decay, step):
    if decay == 'exp':
        return var_decay ** step
    elif decay == 'lin':
        return var_decay / (step + var_decay)
    else:
        return var_decay / (step + np.sqrt(var_decay))


def run_experiment(img_size, device, prior_var, var_decay, n_steps, sample_size, kappa, gamma,
                   decay='exp', plot=True, empir_cov=False):
    n = img_size ** 2  # dimensionality of barycenter
    dtype = torch.float32

    cost_mat = get_cost_matrix(img_size, device, dtype=dtype)
    cs, r_opt = get_data_and_solution(device, size=img_size, column_interval=1 if img_size <= 5 else 2)
    m = cs.shape[0]

    def objective(sample):
        # 'sample' has size (M, 2*m*n), where M is sample size or 1
        M = sample.shape[0]

        lambdas = [sample[:, i * n:(i + 1) * n] for i in range(m)]
        lambdas = torch.stack(lambdas, dim=0)  # (m, M, n)

        mus = [sample[:, (m+i)*n:(m+i+1)*n] for i in range(m)]
        mus = torch.stack(mus, dim=0)  # (m, M, n)

        val1 = (lambdas.sum(dim=0)) / kappa  # (M, n)
        term1 = -kappa * torch.logsumexp(val1, dim=-1)  # (M,)

        val2 = lambdas * cs.unsqueeze(1)  # (m, M, n)
        term2 = -val2.sum(dim=(0, 2))  # (M,)

        mat = cost_mat + lambdas.unsqueeze(3).expand(-1, -1, -1, n)\
                       + mus.unsqueeze(2).expand(-1, -1, n, -1)  # (m, M, n, n)

        mat = mat.reshape(m, M, -1)  # (m, M, n^2)
        term3 = -gamma * torch.logsumexp(mat, dim=-1).sum(dim=0)  # (M,)

        obj_val = term1 + term2 + term3
        return obj_val

    def get_info(z):
        # 'z' has size (4n,)
        lambda_sum = z[:n] + z[n:2*n]  # (n,)
        r = torch.softmax(lambda_sum, dim=-1)  # (n,)

        acc_r = norm_sq(r - r_opt)  # distance between current approximation and true barycenter
        objective_val = -objective(z.unsqueeze(0))
        return r, objective_val, acc_r

    torch.manual_seed(42)
    z_prior = torch.normal(torch.zeros(2 * m * n, device=device), 10.)

    prior_cov = prior_var * torch.eye(2 * m * n, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    if empir_cov:
        def recalculate_cov(old_cov, sample, step, weights):
            matrix = torch.cov(sample.T, aweights=weights) if empir_cov else prior_cov
            diag = torch.min(torch.diag(matrix))
            factor = get_factor(decay, var_decay, step)
            return factor * (matrix / diag)
    else:
        def recalculate_cov(old_cov, sample, step, weights):
            factor = get_factor(decay, var_decay, step)
            return factor * prior_cov

    trajectory = algorithm(z_prior, prior_cov, get_sample, n_steps, objective,
                           recalculate_cov, seed=0, get_info=get_info)

    if plot:
        # Names and indices of elements of the list returned by 'get_info'
        info_names = [{'Objective': 1}, {'Accuracy of r': 2}]
        n_cols = 6
        img_name = f"dual_samples_{sample_size}_var_{prior_var}_decay_{var_decay}"

        plot_trajectory(trajectory, n_cols, img_size, img_name, info_names)
    else:
        return sum([info[2] for info in trajectory[-5:]]) / 5, trajectory


def seven_by_seven():
    img_size = 7
    device = 'cpu'
    n_steps = 50

    # Hyperparameters
    sample_size = 2 ** 12
    prior_var = 32.
    var_decay = 3
    kappa = 0.01

    gamma = kappa
    run_experiment(img_size, device, prior_var, var_decay, n_steps, sample_size,
                   kappa, gamma, decay='sqrt', empir_cov=False)


def best_params():
    img_size = 3
    device = 'cpu'
    n_steps = 30

    # Hyperparameters
    sample_size = 2**10
    prior_var = 2. ** 4
    var_decay = 0.9  # ** 2
    kappa = 0.02

    gamma = kappa
    run_experiment(img_size, device, prior_var, var_decay, n_steps, sample_size, kappa, gamma)


def grid_search():
    device = 'cpu'
    n_steps = 30
    img_size = 3

    # Hyperparameters
    sample_sizes = [8 ** i for i in range(2, 5)]
    X = list(range(1, 8))
    prior_vars = [2. ** i for i in X]
    Y = list(range(5))
    var_decays = [0.9 ** i for i in Y]
    kappas = [0.01, 0.03, 0.1]

    n_cols = 6
    info_names = [{'Objective': 1}, {'Accuracy of r': 2}]

    for sample_size in sample_sizes:
        print("===== SAMPLE SIZE:", sample_size, "=====")
        # best_acc = 10. ** 6
        # best_hyperpar = None
        # best_traj = None
        Zs = []
        for kappa in kappas:
            print("--- KAPPA:", kappa, '---')
            gamma = kappa
            acuracies = np.zeros((len(var_decays), len(prior_vars)))
            for x, prior_var in enumerate(prior_vars):
                print("   prior_var:", prior_var)
                for y, var_decay in enumerate(var_decays):
                    print("      var_decay:", var_decay)
                    start = time.time()
                    acc_r, traj = run_experiment(img_size, device, prior_var, var_decay, n_steps, sample_size, kappa, gamma, plot=False)
                    # print(f"      exper. took {time.time() - start:.2f} s")
                    print(f"         accuracy {acc_r}")
                    acuracies[y, x] = acc_r
                    # if acc_r < best_acc:
                    #     best_acc = acc_r
                    #     best_hyperpar = (prior_var, var_decay, kappa)
                    #     best_traj = traj
                    # else:
                    del traj
                    gc.collect()
            Zs.append(acuracies)
        plot3d(X, Y, Zs, kappas, sample_size)

        # print(f"Best prior_var: {best_hyperpar[0]}, best var_decay: {best_hyperpar[1]}")
        # img_name = f"dual_samples_{sample_size}_var_{best_hyperpar[0]}_decay_{best_hyperpar[1]}_kappa_{best_hyperpar[2]}"
        # plot_trajectory(best_traj, n_cols, img_size, img_name, info_names)


if __name__ == "__main__":
    # grid_search()
    best_params()
    # seven_by_seven()
