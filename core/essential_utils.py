import torch
import numpy as np
import itertools


def get_cost_mat(im_sz, device, dtype=torch.float32):
    partition = torch.linspace(0, 1, im_sz)
    couples = np.array(np.meshgrid(partition, partition)).T.reshape(-1, 2)
    x = np.array(list(itertools.product(couples, repeat=2)))
    x = torch.tensor(x, dtype=dtype, device=device)
    a = x[:, 0]
    b = x[:, 1]
    C = torch.linalg.norm(a - b, axis=1) ** 2
    return C.reshape((im_sz**2, -1))


def get_c_concave(phi, cost_mat):
    # 'phi' has size (M, m*n), where M is sample size or 1
    M = phi.shape[0]
    n = cost_mat.shape[0]
    m = phi.shape[1] // n
    phi_c, _ = (cost_mat - phi.reshape(M, m, n, 1)).min(dim=2)  # (M, m, n)
    return phi_c


def get_sample_generator(prior_mean, n_batches, prior_std, verbose=False):
    def sample_generator():
        for i in range(n_batches):
            if verbose:
                print(f"sampling batch {i}")
            torch.manual_seed(i)
            yield torch.normal(prior_mean, prior_std)

    return sample_generator
