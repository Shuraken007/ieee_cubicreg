import numpy as np
node_type = np.array([2, 2, 2, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1])
a = np.where(node_type == 3)[0][0]
print(a)