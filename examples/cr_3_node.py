from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rand
import src.cubic_reg
import src.createdata
import src.easy_print as ep
import time

if __name__ == '__main__':
    # Choose a function to run it on, and a method to use (original cubic reg or adaptive cubic reg)
    # Function choices: 'bimodal', 'simple', 'quadratic'
    # Method choices: 'adaptive', 'original'
    # If you choose method='adaptive', you can choose hessian updates from 'broyden', 'rank_one', and 'exact'.
    num_node = 3
    ballance_node = 1
    Z = np.array([
        [0       , 10 + 20j, 15+30j],
        [10 + 20j,        0, 10+25j],
        [15 + 30j, 10 + 25j,      0]
    ])
    T = np.array([
        [0, 2, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
#    Z = [Z, None, T]
    S = np.array([28.8680 - 700j, -46.1880 - 23.094j]);
    U_0 = 115
    f, grad, hess = src.createdata.generate_cubic_system(S, Z, num_node, ballance_node, U_0)
    u = np.ones((1, num_node-1))*110
    v = np.ones((1, num_node - 1)) * 0
    x0 = np.concatenate((u, v)).ravel()
    cr = src.cubic_reg.CubicRegularization(x0, f, grad, hess, conv_tol=1e-10)

steptype = 'first'
m_file = 'cr_3_node_None' + steptype + '.txt'

easy = ep.easy_print()
easy.add_phase('iter', 'grad', 'value', 'M', 'e_min', 'time', 'vector',
               phase='main_iter', file=m_file, sep=' | ',
               iter='{:>6}',
               grad='{:5.3e}',
               value='{:5.3e}',
               M='{:11.5f}',
               e_min='{:11.5f}',
               time='{:+f}',
               vector=np.set_printoptions(formatter={'float': '{:7.3f}'.format}, linewidth=10000)
               )
easy.add_phase('iter', 'grad', 'value', 'M', 'e_min', 'time', 'vector',
               phase='sub_M', file='cr_3_node_mode_' + steptype + '.txt', sep=' | ',
               check_phase='main_iter',
               M='{:11.5f}',
               )
start = time.time()
cr = src.cubic_reg.CubicRegularization(x0, f, None, None, conv_tol=1e-10, L0=0.1, print_data=easy, stepmin = steptype, epsilon=0.0001)
x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
end = time.time()
easy.print_head({'vector': {'form': 'all_columns', 'pattern': 'x[{:d}]', 'length': len(x0)}}, phase='main_iter')
with open(m_file, 'a') as fi:
    np.set_printoptions(formatter={'float': '{:11.11e}'.format}, linewidth=10000)
    print('\n\n\n', file=fi)
    print('{:<28}'.format('x optimal:'), x_opt.ravel(), file=fi)
    print('{:<28}'.format('S degree:'), S - src.createdata.test_U(x_opt, Z, ballance_node, U_0), file=fi)
    print('{:<28}'.format('eigen vals, hessian(x_opt):'), np.linalg.eig(hess(x_opt))[0], file=fi)
    print('{:<28}'.format('e_min:'), np.linalg.eig(hess(x_opt))[0].min(), file=fi)
    print('{:<28}'.format('total_time:'), end - start, file=fi)