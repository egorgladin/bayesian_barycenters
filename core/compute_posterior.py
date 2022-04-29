import torch
from tqdm import tqdm
from essential_utils import get_c_concave, get_cost_mat, get_sample_generator


def objective_function(sample, cost_mat, cs, kappa):
    # 'sample' has size (M, m*n), where M is sample size or 1
    phi_c = get_c_concave(sample, cost_mat)  # (M, m, n)
    phi_bar = phi_c.sum(dim=1)  # (M, n)

    logsumexp = -kappa * torch.logsumexp(-phi_bar / kappa, dim=-1)  # (M,)
    inner_prod = (phi_c * cs).sum(dim=(1, 2))  # (M,)
    return logsumexp + inner_prod


def get_posterior_mean_cov(sample_generator, objective, temperature, n_batches=None):
    """
    Calculate weighted empirical mean and covariance matrix.

    :param sample_generator: generator function which returns batches of samples
    :param objective: function that receives a batch of samples, i.e., tensor of shape (batch_size, variable_dimension),
        and returns tensor of objective values of shape (batch_size,)
    :param temperature: temperature number used for calculating weights
    :param n_batches: number of batches returned by sample_generator (needed for a progress bar)
    :return: (mean, cov)
    """
    objective_values = []
    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        objective_values.append(objective(batch))

    batch_sizes = [len(b) for b in objective_values]
    weights = torch.softmax(temperature * torch.cat(objective_values), dim=-1)

    start_idx = 0
    for i, batch in enumerate(tqdm(sample_generator(), total=n_batches)):
        factor = weights[start_idx:start_idx+batch_sizes[i]].sum()
        batch_weights = torch.softmax(temperature * objective_values[i], dim=-1)

        if i == 0:
            result_mean = factor * (batch_weights @ batch)
            result_cov = factor * ((batch_weights.unsqueeze(1) * batch).unsqueeze(2) @ batch.unsqueeze(1)).sum(dim=0)
        else:
            result_mean += factor * (batch_weights @ batch)
            result_cov += factor * ((batch_weights.unsqueeze(1) * batch).unsqueeze(2) @ batch.unsqueeze(1)).sum(dim=0)
        start_idx += batch_sizes[i]

    result_cov -= torch.outer(result_mean, result_mean)
    return result_mean, result_cov


def main(data_path, mean_path, sample_size, n_batches, prior_var, temperature, kappa, save_path, device='cpu'):
    img_sz = 28
    cs = torch.load(data_path, map_location=device)
    dtype = cs.dtype
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)

    potentials = torch.load(mean_path, map_location=device)
    if potentials.requires_grad:
        potentials = potentials.detach()

    objective = lambda sample: objective_function(sample, cost_mat, cs, kappa)

    increment = sample_size // n_batches
    prior_std = torch.sqrt(torch.tensor(prior_var, device=device, dtype=dtype))

    prior_mean = potentials.reshape(1, -1).expand(increment, -1)
    sample_generator = get_sample_generator(prior_mean, n_batches, prior_std)

    result_mean, result_cov = get_posterior_mean_cov(sample_generator, objective, temperature, n_batches=n_batches)
    torch.save(result_mean, save_path + 'mean.pt')
    torch.save(result_cov, save_path + 'cov.pt')


if __name__ == '__main__':
    pass
