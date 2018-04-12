import numpy as np
a = np.array([1, 1])
b = np.array([2, 3])
c = np.array([[1, 2], [3, 4]])
d = np.array([[1, 1], [1, 1]])
D = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 0, -1, 0],
              [0, 1, 0, -1]])
'''
# произведение векторов
print(a * b)  # скалярное произведение (a, b)
print(a.dot(b))  # произведение вектора на столбец
print(a @ b)  # то же
print(np.matmul(a, b))  # то же
# вектор на матрицу
print(b * c)  # скалярное умножение
print(b.dot(c))  # умножение вектора на матрицу слева
print(b@c)
print(np.matmul(b, c))
print(c.dot(b))  # умножение вектора на матрицу справа
print(c@b)
print(np.matmul(c, b))
# матрица на матрицу
print(c * d)  # скалярное поэлементное умножение
print(c.dot(d))
print(c@d)
print(d.dot(c))
print(d@c)
# да, всё в той же логике
print(D@np.array([1, 1, 1, 1]))
'''
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])
d = np.array([[13, 14], [15, 16]])
print(a)
print(np.bmat([[a, b], [c, d]]).A)
