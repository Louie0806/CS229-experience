import numpy as np

A = np.array([5, 1, 6, -1])
A = A.reshape(2, 2)
B = np.linalg.inv(A)
print(B)
