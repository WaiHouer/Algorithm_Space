import numpy as np


m = np.zeros((3,3))

E = np.zeros((3, 60))
for i in range(3):
    E[i][0] = 100
print(E,E[i][0])

