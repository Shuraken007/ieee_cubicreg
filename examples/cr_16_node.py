from __future__ import print_function

import numpy as np
# import numpy.random as rand
import sys
sys.path.append('D:\\Development\\Projects\\ieee_cubicreg')
import src.cubic_reg
import src.createdata
import src.easy_print as ep
import time

if __name__ == '__main__':

  num_node = 16
  ballance_node = 16 - 1  # - 1 because of array indexing starting from 0
  name_node = np.array([40, 42, 73, 136, 186, 187, 206, 207, 209, 212, 214, 220, 222, 228, 397, 700])
  node_from = np.array([397, 73, 186, 206, 206, 206, 206, 206, 209, 209, 212, 214, 214, 222, 222, 228])
  node_to = np.array([207, 212, 206, 40, 42, 220, 228, 700, 187, 207, 207, 212, 222, 136, 187, 73])
  r = np.array([7.74, 2.2, 2.4, 0.6, 0.6, 6.2, 0.8, 4.12, 2.2, 1.2, 0.4, 3.5, 3.6, 7.8, 1.9, 0.3])
  x = np.array([38.41, 13.2, 12.6, 3.1, 3.1, 31.6, 4.7, 23.14, 11.6, 6.2, 2.4, 17, 19.3, 31.2, 10, 1.9])
  b = np.array([-262.99999, - 99, - 83, - 21, - 21, - 210, - 31, - 152.4, - 78, - 40, - 16, - 111, - 130.99999, - 204, - 69, - 13]) * 10**(-6)

  Z = np.zeros((16, 16), dtype='complex')
  bg = np.zeros((16, 16), dtype='complex')

  for i in range(0, num_node):
    Z[np.where(name_node == node_from[i])[0], np.where(name_node == node_to[i])[0]] = r[i] + x[i] * 1.j
    Z[np.where(name_node == node_to[i])[0], np.where(name_node == node_from[i])[0]] = r[i] + x[i] * 1.j
    bg[np.where(name_node == node_from[i])[0], np.where(name_node == node_to[i])[0]] = b[i]
    bg[np.where(name_node == node_to[i])[0], np.where(name_node == node_from[i])[0]] = b[i]
  bg = bg * 1.j
  # S = np.rand m.sample(num_node-1)*(100+80j)
  S = np.array([0.78203, 4.012712, -6.45068, -6.45086, -5.59356, -4.30349, 10.7697, -3.49422, -1.86383, 59.4273, -0.39498,
                -3.81818, -0.16991, -3.15909, 14.52273]) - 400j
  # S = np.ones((1, 15))
  U_0 = 226.04613
  Z = {'Z': Z, 'bg': bg}
  f, grad, hess = src.createdata.generate_cubic_system(S, Z, num_node, ballance_node, U_0)
  u = np.ones(num_node - 1) * (220)
  v = np.ones(num_node - 1) * 0
  x0 = np.concatenate((u, v))

  steptype = 'base+diff'
  m_file = '..\\result\\' + 'cr_16_node_' + steptype + '.txt'

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
                 phase='sub_M', file=m_file, sep=' | ',
                 check_phase='main_iter',
                 M='{:11.5f}',
                 )
  easy.add_phase('S',
                 phase='S', file=m_file, sep=' | ',
                 check_phase='main_iter',
                 S=np.set_printoptions(formatter={'float': '{:7.3f}'.format}, linewidth=10000)
                 )

  np.set_printoptions(formatter={'float': '{:7.3f}'.format}, linewidth=10000)
  easy.file_list[m_file] = 1
  open(m_file, 'w')

  with open(m_file, 'a') as fi:
    print('\n\n\n', file=fi)
    print('S = ', S, file=fi)
  f, grad, hess = src.createdata.generate_cubic_system(S, Z, num_node, ballance_node, U_0)
  start = time.time()
  cr = src.cubic_reg.CubicRegularization(x0, f, grad, hess, conv_tol=1e-8, L0=0.2, print_data=easy,
                                         stepmin=steptype, epsilon=0.0001)
  x_opt, intermediate_points, n_iter, flag = cr.cubic_reg()
  end = time.time()
  with open(m_file, 'a') as fi:
    np.set_printoptions(formatter={'float': '{:11.11e}'.format}, linewidth=10000)
    print('\n', file=fi)
    print('{:<28}'.format('x optimal:'), x_opt.ravel(), file=fi)
    print('{:<28}'.format('S degree:'), S - src.createdata.test_U(x_opt, Z, ballance_node, U_0), file=fi)
    print('{:<28}'.format('eigen vals, hessian(x_opt):'), np.linalg.eig(hess(x_opt))[0], file=fi)
    print('{:<28}'.format('e_min:'), np.linalg.eig(hess(x_opt))[0].min(), file=fi)
    print('{:<28}'.format('total_time:'), end - start, file=fi)

  if i == 1:
    easy.print_head({'vector': {'form': 'all_columns', 'pattern': 'x[{:d}]', 'length': len(x0)}},
                    phase='main_iter')
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
                 phase='sub_M', file=m_file, sep=' | ',
                 check_phase='main_iter',
                 M='{:11.5f}',
                 )
  easy.add_phase('S',
                 phase='S', file=m_file, sep=' | ',
                 check_phase='main_iter',
                 S=np.set_printoptions(formatter={'float': '{:7.3f}'.format}, linewidth=10000)
                 )
