import ot
import torch

from utils import get_cost_mat, show_barycenters
from kantorovich_dual import get_c_concave

from tqdm.contrib.itertools import product


def visualize_potentials(img_sz, cs, r_opt, cost_mat, kappa, image_name, use_emd=True,
                         calc_poten=True, poten_path=None, sinkhorn_reg=0.01):
    if calc_poten:
        u = []
        v = []
        for c in cs:
            res = ot.emd(r_opt, c, cost_mat, log=True)[1] if use_emd\
                else ot.sinkhorn(r_opt, c, cost_mat, sinkhorn_reg, log=True, stopThr=1e-10, numItermax=50000)[1]
            u.append(res['u'])
            v.append(res['v'])

        u, v = torch.cat(u), torch.cat(v)
        torch.save(u, poten_path + '_u.pt')
        torch.save(v, poten_path + '_v.pt')
    else:
        u = torch.load(poten_path + '_u.pt', map_location=cs.device)
        v = torch.load(poten_path + '_v.pt', map_location=cs.device)

    u, v = u.unsqueeze(0), v.unsqueeze(0)
    u_c, u = get_c_concave(u, cost_mat, double=True)  # (1, m, n)
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
    cost_mat = get_cost_mat(img_sz, device, dtype=dtype)
    cs = torch.load(folder + 'digits.pt', map_location=device)
    r = torch.load(folder + 'barycenter.pt', map_location=device)

    # images = [c for c in cs] + [r]
    # titles = [f'c{i}' for i in range(cs.shape[0])] + ['ot.barycenter']
    # show_barycenters(images, img_sz, 'digit_inputs', use_softmax=False, iterations=titles, scaling='partial')

    kappas = [0.1, 0.3, 0.6]
    regs = [0.1, 0.5, 1., 2.]

    for kappa, reg in product(kappas, regs):
        calc_poten = (kappa == kappas[0])
        image_name = f'digits_sinkh_kappa_{kappa}_reg_{reg}'
        visualize_potentials(img_sz, cs, r, cost_mat, kappa, image_name, use_emd=False,
                             calc_poten=calc_poten, poten_path=folder+f'sinhk_reg{reg}', sinkhorn_reg=reg)


if __name__ == "__main__":
    digits_problem()
