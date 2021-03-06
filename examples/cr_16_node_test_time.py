from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import src.cubic_reg
import src.createdata
import src.easy_print as ep
import time

if __name__ == '__main__':

    num_node = 16
    ballance_node = 16 - 1
    name_node = np.array([40, 42, 73, 136, 186, 187, 206,
                          207, 209, 212, 214, 220, 222, 228, 397, 700])
    node_from = np.array([397, 73, 186, 206, 206, 206, 206,
                          206, 209, 209, 212, 214, 214, 222, 222, 228])
    node_to = np.array([207, 212, 206, 40, 42, 220, 228, 700,
                        187, 207, 207, 212, 222, 136, 187, 73])
    r = np.array([7.74, 2.2, 2.4, 0.6, 0.6, 6.2, 0.8, 4.12,
                  2.2, 1.2, 0.4, 3.5, 3.6, 7.8, 1.9, 0.3])
    x = np.array([38.41, 13.2, 12.6, 3.1, 3.1, 31.6, 4.7, 23.14,
                  11.6, 6.2, 2.4, 17, 19.3, 31.2, 10, 1.9])
    b = np.array([-262.99999, - 99, - 83, - 21, - 21, - 210, - 31, - 152.4, -
                  78, - 40, - 16, - 111, - 130.99999, - 204, - 69, - 13]) * 10**(-6)
    Z = np.zeros((16, 16), dtype='complex')
    B = np.zeros((16, 16), dtype='complex')
    for i in range(0, num_node):
        Z[np.where(name_node == node_from[i])[0], np.where(
            name_node == node_to[i])[0]] = r[i] + x[i] * 1.j
        Z[np.where(name_node == node_to[i])[0], np.where(
            name_node == node_from[i])[0]] = r[i] + x[i] * 1.j
        B[np.where(name_node == node_from[i])[0], np.where(
            name_node == node_to[i])[0]] = b[i]
        B[np.where(name_node == node_to[i])[0], np.where(
            name_node == node_from[i])[0]] = b[i]
    B = B * 1.j
    # S = np.array([0.81182924 + 0.64946339j,  1.05736861 + 0.84589489j,  0.25771367 + 0.20617094j,  0.30024930 + 0.24019944j,
    # 1.19543570 + 0.95634856j,  0.12965216 + 0.10372172j,  0.61159000 + 0.489272j,    0.70007139 + 0.56005711j,
    # 1.61938380 + 1.29550704j,  0.20625151 + 0.1650012j,   0.45190501 + 0.36152401j,  1.24147592 + 0.99318073j,
    # 0.90140092 + 0.72112073j,  1.47609373 + 1.18087499j,  0.47029471 + 0.37623576j])
    # S = -S
    # S = np.random.sample(num_node-1)*(1+0.8j)
    base_S = np.array([0.78203, 0.032712, -0.45068, -0.45086, -0.59356, -0.30349, 0.7697, -3.49422, -1.86383, 0.4273, -0.39498,
                       -3.81818, -0.16991, -0.15909, 0.52273], dtype='complex')
    #S = np.ones(15, dtype='complex')
    U_0 = 226.04613
    # Z = [Z, B]
    u = np.ones((1, num_node - 1)) * (220)
    v = np.ones((1, num_node - 1)) * 0
    x0 = np.concatenate((u, v)).ravel()

    file_base_name = 'test_time\\test_time'
    methods = ['base', 'diff', 'base+diff', 'bifind']
    mods = {'+': 70, '-': -70, '+j': 70j, '-j': -70j}

    np.set_printoptions(formatter={'float': '{:7.3f}'.format}, linewidth=10000)

    for k, v in mods.items():
        S = base_S.copy()
        print(k, np.linalg.norm(S))
        res_file = file_base_name + k + '.txt'
        open(res_file, 'w')
        e = []
        for i, m in enumerate(methods):
            e.append(ep.easy_print())
            e[i].add_phase('iter', 'grad', 'value', 'e_min', 'S',
                           phase='last_phase', file='test_time\\table_'+m+k+'.txt', sep=' | ',
                           iter='{:>6}',
                           grad='{:5.3e}',
                           value='{:5.3e}',
                           e_min='{:11.5f}',
                           S='{:5.3e}'
                           )

        for i in range(1, 30):
            res_time = []
            start = []
            res_string = ''
            f, grad, hess = src.createdata.generate_cubic_system(
                S, Z, num_node, ballance_node, U_0)
            for i, m in enumerate(methods):
                start = time.time()
                cr = src.cubic_reg.CubicRegularization(
                    x0, f, grad, hess, conv_tol=1e-5, L0=0.01, print_data=e[i], stepmin=m, S=S)
                x_opt, intermediate_points, n_iter, fla g = cr.cubic_reg()
                end = time.time()
                res_time.append(end-start)
                if flag == 1:
                    res_time[i] = prev_time[i]
                res_string += '{:f}'.format(res_time[i]) + ' '

            with open(res_file, 'a') as fi:
                print(res_string +
                      '{:f}'.format(np.linalg.norm(np.abs(S))) + '\n', file=fi)

            S += v
            prev_time = res_time

        for i, f in enumerate(methods):
            e[i].print_head(phase='last_phase')
