import torch

from kantorovich_dual import objective_function
from experiment_barycenter import get_cost_matrix, get_data_and_solution
from utils import get_sampler
from algorithm import algorithm


def get_posterior_mean(sample_generator, objective, temperature):
    batch_means = []
    objective_values = []
    for batch in sample_generator():  # batch shape: (batch_size, n)
        batch_values = objective(batch)  # (batch_size,)
        objective_values.append(batch_values)
        batch_weights = torch.softmax(temperature * batch_values, dim=-1)  # (batch_size,)
        batch_means.append(batch_weights @ batch)
        del batch

    batch_sizes = [len(b) for b in objective_values]
    weights = torch.softmax(temperature * torch.cat(objective_values), dim=-1)  # (sample_size,)
    del objective_values

    result = torch.zeros_like(batch_means[0])
    start_idx = 0
    for batch_size, batch_mean in zip(batch_sizes, batch_means):
        factor = weights[start_idx:start_idx+batch_size].sum()
        result += factor * batch_mean
        start_idx += batch_size

    return result


def check_mean():
    img_size = 3
    device = 'cpu'
    kappa = 1. / 30
    temp = 30.

    sample_size = 1000
    n = img_size ** 2  # dimensionality of barycenter
    dtype = torch.float64

    cost_mat = get_cost_matrix(img_size, device, dtype=dtype)
    cs, r_opt = get_data_and_solution(device, dtype=dtype, size=img_size, column_interval=(1 if img_size == 3 else 2))
    m = cs.shape[0]

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa, double_conjugate=False)[0]

    z_prior = -torch.ones(m * n, device=device, dtype=dtype)
    prior_cov = torch.eye(m * n, dtype=dtype, device=device)
    get_sample = get_sampler(sample_size)

    def recalculate_cov(old_cov, x, y, z):
        return old_cov

    n_steps = 1
    true_posterior_mean = algorithm(z_prior, prior_cov, get_sample, n_steps, objective,
                                    recalculate_cov, seed=0, temperature=temp)[-1]

    def sample_generator():
        sample = get_sample(z_prior, prior_cov, 0)
        yield sample[:sample_size//2]
        yield sample[sample_size//2:]

    batch_posterior_mean = get_posterior_mean(sample_generator, objective, temp)
    error = torch.norm(true_posterior_mean - batch_posterior_mean) / torch.norm(true_posterior_mean)
    print(f"Relative error: {100 * error:.2e}%")


if __name__ == '__main__':
    check_mean()
