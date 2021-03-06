import numpy as np
import src.cubic_reg
import src.createdata
import src.easy_print as ep
import os
import sys
import time

def pretty(d, log, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key), file = log)
      if isinstance(value, dict):
         pretty(value, file, indent+1)
      else:
        if isinstance(value, str):
          value = value.split(';')
        else:
          value = [value]
        for v in value:
          print('\t' * (indent+1) + str(v), file = log)

if __name__ == '__main__':

  log = open('log.txt', 'w')
  print('>>>>>>>>>>>sys.path<<<<<<<<<<<<<<\n\n\n', file = log)
  for p in sys.path:
      print(p, file = log)
  print('\n\n\n>>>>>>>>>>>os.environ<<<<<<<<<<<<<<\n\n\n', file = log)
  pretty(os.environ, log)
  num_node = 3
  ballance_node = 3 - 1
  Z = np.array([
      [0, 10 + 20j, 15 + 30j],
      [10 + 20j, 0, 10 + 25j],
      [15 + 30j, 10 + 25j, 0]
  ])
  T = np.array([
      [0, 2, 0],
      [0, 0, 0],
      [0, 0, 0]
  ])
  # Z = [Z, None, T]
  S = np.array([28.8675 + 17.3205j, -46.1880 - 23.094j])
  U_0 = 115
  f, grad, hess = src.createdata.generate_cubic_system(S, Z, num_node, ballance_node, U_0)
  f, grad, hess = src.createdata.generate_polar_system(f, grad, hess)
  u = np.ones(num_node - 1) * 110
  v = np.ones(num_node - 1) * 0
  x0 = np.append(u, v)
  x0_polar = np.append(abs(u + v * 1.j), np.angle(u + v * 1.j))

  steptype = 'base+diff'
  m_file = '..\\result\\' + 'cr_3_node_polar_' + steptype + '.txt'

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
  start = time.time()
  # cr = src.cubic_reg.CubicRegularization(x0, f, grad, hess, conv_tol=1e-10, L0=0.11, print_data=easy, stepmin=steptype, epsilon=0.0001)
  cr = src.cubic_reg.CubicRegularization(x0_polar, f, grad, hess, conv_tol=1e-10, L0=0.11, print_data=easy, stepmin=steptype, epsilon=0.0001)

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
