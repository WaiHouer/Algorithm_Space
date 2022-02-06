import numpy as np


m = np.zeros((3,3))

n = np.ones((3,3))
n[1][1] = 2

print(m+n)

