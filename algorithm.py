import torch


def algorithm(prior_mean, prior_std, n_steps, sample_size, objective, std_decay=None, seed=None):
    """
    Maximize objective by the bayesian algorithm with sampling.

    :param prior_mean: (n,) tensor
    :param prior_std: scalar tensor
    :param n_steps: int
    :param sample_size: int
    :param objective: function taking (m, n) tensor and returning (m,) tensor
    :param std_decay: None or scalar tensor, in the latter case prior_std is multiplied by it at each step
    :param seed: int
    :return: (n,) tensor
    """
    trajectory = [prior_mean.clone()]
    for step in range(n_steps):
        if seed is not None:
            torch.manual_seed(seed + step)
        sample = torch.normal(prior_mean.expand(sample_size, -1), prior_std)  # (sample_size, n)

        objective_values = objective(sample)  # (sample_size,)
        weights = torch.softmax(objective_values, dim=-1)  # (sample_size,)
        prior_mean = weights @ sample
        trajectory.append(prior_mean.clone())

        if std_decay is not None:
            prior_std *= std_decay
    return trajectory
