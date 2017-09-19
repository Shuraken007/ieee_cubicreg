import numpy as np
import src.cubic_reg
import src.createdata
import src.easy_print as ep
import time

if __name__ == '__main__':

  name_node = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
  node_from = np.array([1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 9, 9, 10, 12, 13])
  node_to = np.array([2, 5, 3, 4, 5, 4, 5, 7, 9, 6, 11, 12, 13, 8, 9, 10, 14, 11, 13, 14])
  r = np.array([10.2520, 28.5819, 24.8577, 30.7402, 30.1266, 35.4483, 7.0622, 0.0000, 0.0000, 0.0000, 12.5611, 16.2548, 8.7483,
                0.0000, 0.0000, 4.2069, 16.8103, 10.8511, 29.2167, 22.6055])
  x = np.array([31.3009, 117.9882, 104.7261, 93.2733, 91.9825, 90.4749, 22.2762, 105.8105, 276.2604, 115.8037,
                26.3045, 33.8309, 17.2282, 23.2958, 14.5488, 11.1751, 35.7578, 25.4013, 26.4341, 46.0256])
  b = np.array([0.0998, 0.0930, 0.0828, 0.0643, 0.0654, 0.0242, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, ])

  t = np.array([0, 0, 0, 0, 0, 0, 0, 0.511247, 0.515996, 0.536481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  node_type = np.array([3, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1])
  vzd = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1.43667, 0, 0, 0, 0, 0, ]) * 10 ** -3
  num_node = node_type.size
  ballance_node = np.where(node_type == 3)[0]

  Z = np.zeros((num_node, num_node), dtype='complex')
  bg = np.zeros((num_node, num_node), dtype='complex')
  T = np.zeros((num_node, num_node), dtype='complex')

  for i in range(0, num_node):
    Z[np.where(name_node == node_from[i])[0], np.where(name_node == node_to[i])[0]] = r[i] + x[i] * 1.j
    Z[np.where(name_node == node_to[i])[0], np.where(name_node == node_from[i])[0]] = r[i] + x[i] * 1.j
    bg[np.where(name_node == node_from[i])[0], np.where(name_node == node_to[i])[0]] = b[i]
    bg[np.where(name_node == node_to[i])[0], np.where(name_node == node_from[i])[0]] = b[i]
    T[np.where(name_node == node_from[i])[0], np.where(name_node == node_to[i])[0]] = t[i]
    T[np.where(name_node == node_to[i])[0], np.where(name_node == node_from[i])[0]] = t[i]
  bg = bg * 1.j
  S = np.array([21.70 - 40.0 + 12.70j, 94.20 + 19.00j, 47.80 - 3.90j, 7.60 + 1.60j, 11.20 + 7.50j, 0 + 0j, 0 + 0j,
                29.50 + 16.60j, 9.00 + 5.80j, 3.50 + 1.80j, 6.10 + 1.60j, 13.50 + 5.80j, 14.90 + 5.00])

  Z = {'Z': Z, 'T': T, 'bg': bg, 'vzd': vzd}

  u = np.array([230.0, 230.0, 230.0, 230.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0, 115.0])
  v = np.ones(num_node - 1) * 0

  U_0 = 230.0

  f, grad, hess = src.createdata.generate_cubic_system(S, Z, num_node, ballance_node, U_0)

  x0 = np.concatenate((u, v)).ravel()

  steptype = 'base'
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
  cr = src.cubic_reg.CubicRegularization(x0, f, None, None, conv_tol=1e-8, L0=0.1, print_data=easy,
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
