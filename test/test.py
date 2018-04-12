import numpy as np

if __name__ == '__main__':
   num_node = 3

   def v(x): return x[:x.size // 2]

   def th(x): return x[x.size // 2:]

   def convert_p_to_d(x): return np.append(v(x) * np.cos(th(x)), v(x) * np.sin(th(x)))
   x = np.array([7.45484621410e+01, 4.88091190007e+01, - 2.37785840869e+00, - 7.18760105598e-01])
   print(convert_p_to_d(x))
