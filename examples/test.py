import numpy as np
from scipy.optimize import newton

I = np.identity(2)
H = np.array([
    [100, 100],
    [100, 100]
])
g = np.array([5, 40])
M = 2.4
x = lambda r: np.linalg.inv(H + M * r * I / 2).dot(g)
F_n = lambda r: r * r - np.linalg.norm(x(r)) ** 2
diff_x = lambda r: -M / 2 * (np.linalg.inv(H + M * r * I / 2)).dot(x(r));
diff_F = lambda r: 2 * r - 2 * np.sum(x(r).dot(diff_x(r)));
result = newton(F_n, 1, fprime=diff_F, args=(), tol=1.48e-08, maxiter=1000)
print(result)
print(result)
r_n = (2*(g - H.dot(x(result)))/(M*I.dot(x(result))))[1]
print(r_n)

#print(2 * (g + H.dot(result)))
#print(M * I.dot(result))
#r_n = 2 * (g + H.dot(result)) / (M * I.dot(result))
#print('x^-1 = ', r_n)
# print("diff_res")
# print(result)