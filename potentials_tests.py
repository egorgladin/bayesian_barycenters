import ot
import torch

from experiment_barycenter import get_cost_matrix, get_data_and_solution
from utils import replace_zeros, show_barycenters
from kantorovich_dual import get_c_concave


def visualize_potentials(img_sz, cs, r_opt, cost_mat, kappa, image_name, reverse_order=False):
    # Direct order
    u = []
    v = []
    for c in cs:
        res = ot.emd(c, r_opt, cost_mat, log=True)[1] if reverse_order\
            else ot.emd(r_opt, c, cost_mat, log=True)[1]
        u.append(res['u'])
        v.append(res['v'])

    u = torch.cat(u).unsqueeze(0)  # (1, m*n)
    u_c, u = get_c_concave(u, cost_mat, double=True)  # (1, m, n)
    v = torch.cat(v).unsqueeze(0)
    v_c, v = get_c_concave(v, cost_mat, double=True)  # (1, m, n)

    u_sum = u.squeeze(0).sum(dim=0)  # (n,)
    v_sum = v.squeeze(0).sum(dim=0)  # (n,)
    u_c_sum = u_c.squeeze(0).sum(dim=0)  # (n,)
    v_c_sum = v_c.squeeze(0).sum(dim=0)  # (n,)
    potentials = [u_sum, v_sum, u_c_sum, v_c_sum]

    images = []
    # Without minus
    for potential in potentials:
        bary = torch.softmax(potential / kappa, dim=0)
        images.append(bary.clone())

    # With minus
    for potential in potentials:
        bary = torch.softmax(-potential / kappa, dim=0)
        images.append(bary.clone())
    
    potentials_names = ['u', 'v', 'u_c', 'v_c']
    titles = [f'Softmax({poten} / {kappa})' for poten in potentials_names]
    titles += [f'Softmax(-{poten} / {kappa})' for poten in potentials_names]
    show_barycenters(images, img_sz, image_name, use_softmax=False, iterations=titles, scaling='partial')


def digits_problem():
    dtype = torch.float64
    folder = 'digit_experiment/'
    img_sz = 8
    device = 'cpu'
    cost_mat = get_cost_matrix(img_sz, device, dtype=dtype)
    cs = torch.load(folder + 'digits.pt', map_location=device)
    r = torch.load(folder + 'barycenter.pt', map_location=device)

    images = [c for c in cs] + [r]
    titles = [f'c{i}' for i in range(cs.shape[0])] + ['ot.barycenter']
    show_barycenters(images, img_sz, 'digit_inputs', use_softmax=False, iterations=titles, scaling='partial')

    for kappa in [1., 0.1, 0.01]:
        image_name = f'digits_direct_kappa_{kappa}'
        visualize_potentials(img_sz, cs, r, cost_mat, kappa, image_name)


def toy_problem():
    device = 'cpu'
    img_sz = 3
    dtype = torch.float64
    cost_mat = get_cost_matrix(img_sz, device, dtype=dtype)
    cs, r_opt = get_data_and_solution(device, dtype=dtype, size=img_sz)
    for kappa in [1., 0.01]:
        image_name = f'3x3_direct_kappa_{kappa}'
        visualize_potentials(img_sz, cs, r_opt, cost_mat, kappa, image_name)


if __name__ == "__main__":
    # kappa = 0.01
    digits_problem()
    # toy_problem()
