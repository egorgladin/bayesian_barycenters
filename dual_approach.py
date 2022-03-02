import torch
import time
import gc

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np

from algorithm import algorithm
from experiment_barycenter import get_cost_matrix, get_data_and_solution
from utils import plot_trajectory, norm_sq, get_sampler



def run_experiment(prior_var, var_decay, n_steps, sample_size, kappa, gamma, decay='exp', plot=True):
    img_size = 3
    n = img_size ** 2  # dimensionality of barycenter
    device = 'cpu'
    dtype = torch.float32

    cost_mat = get_cost_matrix(img_size, device, dtype=dtype)
    cs, r_opt = get_data_and_solution(device)

    def objective(sample):
        # 'sample' has size (M, 4n), where M is sample size or 1
        M = sample.shape[0]

        lambda1 = sample[:, :n]  # (M, n)
        lambda2 = sample[:, n:2*n]  # (M, n)
        mu1 = sample[:, 2*n:3*n]  # (M, n)
        mu2 = sample[:, 3*n:]  # (M, n)

        val1 = (lambda1 + lambda2) / kappa  # (M, n)
        term1 = -kappa * torch.logsumexp(val1, dim=-1)  # (M,)

        val2 = torch.stack((lambda1, lambda2), dim=1) * cs  # (M, 2, n)
        term2 = -val2.sum(dim=(1, 2))  # (M,)

        mat1 = cost_mat + lambda1.reshape(M, n, 1).expand(-1, -1, n)\
                        + mu1.reshape(M, 1, n).expand(-1, n, -1)  # (M, n, n)

        mat2 = cost_mat + lambda2.reshape(M, n, 1).expand(-1, -1, n)\
                        + mu2.reshape(M, 1, n).expand(-1, n, -1)  # (M, n, n)

        mat2 = mat2.reshape(M, -1)  # (M, n^2)
        mat1 = mat1.reshape(M, -1)  # (M, n^2)
        term3 = -gamma * (torch.logsumexp(mat1, dim=-1) + torch.logsumexp(mat2, dim=-1))  # (M,)

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
    z_prior = torch.normal(torch.zeros(4 * n), 10.)

    prior_cov = prior_var * torch.eye(4 * n, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    def recalculate_cov(old_cov, sample, step, weights):
        factor = var_decay ** step if decay == 'exp' else var_decay / (step + var_decay)
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


def plot3d(X, Y, Zs, kappas, sample_size):
    fig = plt.figure(figsize=plt.figaspect(0.33))
    # ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(X, Y)

    def log_tick_formatter_x(val, pos=None):
        return f"{int(2**val)}"

    def log_tick_formatter_y(val, pos=None):
        return f"{round(0.9**val, 2)}"

    # Plot the first surface.
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    axes = [ax1, ax2, ax3]
    for i, ax in enumerate(axes):
        Z = Zs[i]
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(0., 1.)
        ax.zaxis.set_major_locator(LinearLocator(6))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        ax.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter_x))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter_y))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel('Initial var')
        ax.set_ylabel('Var decay')

        ax.title.set_text(f"kappa {kappas[i]}")

    fig.suptitle(f'sample_size {sample_size}', fontsize=16)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=8)  # , pad=0.15
    plt.savefig(f"plots/dual3d_sample_size_{sample_size}.png")  #, bbox_inches='tight')


def best_params():
    n_steps = 30

    # Hyperparameters
    sample_size = 64
    prior_var = 2. ** 6
    var_decay = 0.9 ** 4
    kappa = 0.01

    gamma = kappa
    run_experiment(prior_var, var_decay, n_steps, sample_size, kappa, gamma)



def grid_search():
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
                    acc_r, traj = run_experiment(prior_var, var_decay, n_steps, sample_size, kappa, gamma, plot=False)
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
