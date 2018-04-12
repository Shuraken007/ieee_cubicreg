import numpy as np
import src.cubic_reg
import src.createdata
# import src.easy_print as ep
# import time
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from scipy.interpolate import spline
import sympy as sp


def test_f(x):
   z = x[0] + x[1] * 1.j
   return z**2 + 2 * z


def test_g(x):
   return np.array([2 * (x[0] + 1) + 2 * x[1] * 1.j, -2 * x[1] + 2 * (x[0] + 1) * 1.j])


def test_h(x):
   return np.array([[2, 2 * 1.j], [2 * 1.j, -2]])


if __name__ == '__main__':
   x, y, z = sp.symbols('x y z')
   f1 = z**2 + 2 * z
   f1 = f1.subs(z, x + sp.I * y)
   d1 = sp.Matrix([sp.expand(f1.diff(var)) for var in [x, y]])
   h1 = sp.Matrix([[f1.diff(var1).diff(var2) for var1 in [x, y]] for var2 in [x, y]])
   print("f1:", sp.expand(f1.subs([(x, 2), (y, 2)])))
   print("test_f:", test_f([2, 2]))
   print("d1:", d1.subs([(x, 2), (y, 2)]))
   print("test_g:", test_g([2, 2]))
   print("h1:", h1.subs([(x, 2), (y, 2)]))
   print("test_h:", test_h([2, 2]))
   _v, _th = 2.8284271247461903, 0.785398163397
   v, th = sp.symbols('v th')
   f2 = z**2 + 2 * z
   f2 = f2.subs(z, v * (sp.cos(th) + sp.I * sp.sin(th)))
   d2 = sp.Matrix([f2.diff(var) for var in [v, th]])
   h2 = sp.Matrix([[f2.diff(var1).diff(var2) for var1 in [v, th]] for var2 in [v, th]])
   new_f, new_grad, new_hess = src.createdata.generate_polar_system(test_f, test_g, test_h)
   print("f2:", sp.expand(f2.subs([(v, _v), (th, _th)])))
   print("new_f:", new_f(np.array([_v, _th])))
   print("d2:", sp.expand(d2.subs([(v, _v), (th, _th)])))
   print("new_grad:", new_grad(np.array([_v, _th])))
   print("h2:", sp.expand(h2.subs([(v, _v), (th, _th)])))
   print("new_hess:", new_hess(np.array([_v, _th])))
   # print(d.subs([(x, 1), (y, 1)]))
