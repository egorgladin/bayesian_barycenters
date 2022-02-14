import torch
import matplotlib.pyplot as plt
import pickle
import os.path

from algorithm import algorithm


def quadratic_function(A, b, c, sample, is_concave=False):
    """
    Compute f = ||Ax||^2 + <b, x> + c for every row x of sample, or -f if is_concave.

    :param A: (k, n) tensor
    :param b: (n,) tensor
    :param c: scalar tensor
    :param sample: (m, n) tensor
    :param is_concave: if True, return -f(x)
    :return: (m,) tensor
    """
    f = ((sample @ A.T)**2).sum(dim=1) + sample @ b + c
    return -f if is_concave else f


def plot_results(dimensions, sample_sizes, prior_stds, results):
    n_rows, n_cols = len(sample_sizes), len(dimensions)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 12))

    for j, n in enumerate(dimensions):
        for i, sample_size in enumerate(sample_sizes):
            ax = axs[i, j]
            for prior_std in prior_stds:
                distances_from_solution = results.pop(0)
                ax.plot(distances_from_solution, label=f'std={prior_std}')
                ax.set_title(f'n={n}; {sample_size} samples')
            ax.legend()
            ax.set_yscale('log')

    for ax in axs.flat:
        ax.set(xlabel='iterations', ylabel='distance from solution')
    plt.tight_layout()
    plt.savefig(f"experiment_quadratic.png", bbox_inches='tight')


def main():
    device = 'cuda'
    dimensions = [5, 10, 20]
    sample_sizes = [int(2**n / 10) for n in dimensions]
    n_steps = 1000
    prior_stds = [1., 10., 100.]
    std_decay = torch.tensor(0.995, device=device)

    results = []
    for n in dimensions:
        # generate coefficients of quadratic function
        torch.manual_seed(0)
        A = torch.randn(n + 1, n, device=device)
        torch.manual_seed(1)
        b = torch.randn(n, device=device)
        c = torch.tensor(1., device=device)
        objective = lambda sample: quadratic_function(A, b, c, sample, is_concave=True)
        maximizer = torch.linalg.solve(A.T @ A, -b / 2)

        # Ensure the desired distance between initial point and solution
        initial_distance = 1000.
        torch.manual_seed(1)
        direction_to_start = torch.randn(n, device=device)
        direction_to_start /= torch.norm(direction_to_start)
        prior_mean = maximizer + initial_distance * direction_to_start

        for sample_size in sample_sizes:
            print(f"Running algorithm for n={n}, {sample_size} samples")
            for prior_std in prior_stds:
                trajectory = algorithm(prior_mean, prior_std, n_steps, sample_size,
                                       objective, std_decay=std_decay, seed=0)
                distances_from_solution = [torch.norm(v - maximizer).item() for v in trajectory]
                results.append(distances_from_solution)

    with open('experiment_quadratic.pickle', 'wb') as handle:
        pickle.dump((dimensions, sample_sizes, prior_stds, results), handle)


if __name__ == "__main__":
    if os.path.exists('experiment_quadratic.pickle'):
        with open('experiment_quadratic.pickle', 'rb') as handle:
            dimensions, sample_sizes, prior_stds, results = pickle.load(handle)
            plot_results(dimensions, sample_sizes, prior_stds, results)
    else:
        main()
