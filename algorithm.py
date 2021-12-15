"""
Algorithm: maximize objective by the bayesian algorithm with sampling.
"""
import torch
import time


def algorithm(prior_mean, prior_std, n_steps, sample_size, objective,
              std_decay=None, seed=None, get_info=None, track_time=False, hyperparam=None):
    """
    Maximize objective by the bayesian algorithm with sampling.

    :param prior_mean: (d,) tensor
    :param prior_std: scalar tensor
    :param n_steps: int
    :param sample_size: int
    :param objective: function taking (sample_size, d) tensor and returning (sample_size,) tensor
    :param std_decay: None or scalar tensor, in the latter case prior_std is multiplied by it at each step
    :param seed: int
    :param get_info: if None, parameters on each step are stored in trajectory, otherwise it's a function
        output of which is stored in trajectory
    :param track_time: if True, print out duration of each iteration
    :param hyperparam: generator passed to objective
    :return: (d,) tensor
    """
    trajectory = [prior_mean.clone() if get_info is None
                  else get_info(prior_mean)]
    for step in range(n_steps):
        if track_time:
            start = time.time()
        if seed is not None:
            torch.manual_seed(seed + step)
        sample = torch.normal(prior_mean.expand(sample_size, -1), prior_std)  # (sample_size, d)

        objective_values = objective(sample, next(hyperparam)) if hyperparam else objective(sample)  # (sample_size,)
        weights = torch.softmax(objective_values, dim=-1)  # (sample_size,)
        prior_mean = weights @ sample
        trajectory.append(prior_mean.clone() if get_info is None else get_info(prior_mean))

        if std_decay is not None:
            prior_std *= std_decay
        if track_time:
            print(f"Step {step} took {time.time() - start:.2f} s")
    return trajectory
