import numpy as np
from sympy import *

g = np.array([1.1, 0, 3], dtype='float')
g_add = 0.5

H = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype='float')
I = np.identity(np.size(H, 0), dtype='float')
M = 0.3
e, v = np.linalg.eig(H)
e_min = max([0, -np.linalg.eig(H)[0].min()])
g_new = np.linalg.inv(v).dot(g)
for x in np.nditer(g_new, op_flags=['readwrite']):
    if x == 0:
        x[...] = x + g_add

r = symbols('r')
F = r*r
for i in range(g_new.size):
    F = F * (e[i] + r * M / 2) ** 2

for i in range(g_new.size):
    temp = g_new[i] ** 2
    for j in range(g_new.size):
        if j != i:
            temp = temp * (e[j] + r * M / 2) ** 2
    F = F - temp

all_roots = np.roots(Poly(F).all_coeffs())
all_roots = all_roots[all_roots >= e_min]

best_roots = []
best_roots_val = []
best_roots_id = []
index = -1;

point = lambda r: -np.linalg.inv(H + M * r * I / 2).dot(g)
Func_T_M = lambda x: g.dot(x) + (H.dot(x).dot(x))/2 + M/6*np.linalg.norm(x)**3

for i in range(0, all_roots.size):
    if all_roots[i].imag == 0:
        if index == -1:
            print(all_roots[i].real)
            best_roots.append(point(all_roots[i].real))
            best_roots_val.append(Func_T_M(best_roots[0]))
            best_roots_id.append(i);
            index = 0;
        else:
            if np.linalg.norm(best_roots[index] - (point(all_roots[i].real))) > 0.01:
                index = index + 1
                best_roots.append(point(all_roots[i].real))
                best_roots_val.append(Func_T_M(best_roots[index]))
                best_roots_id.append(i)
best_roots_val = np.array(best_roots_val)
print(best_roots)
print(best_roots_val)
print(best_roots_id)
