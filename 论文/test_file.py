import math
import numpy as np


a1 = np.array([[[1,2,3],[1,2,3],[1,2,3]],[[2,4,6],[2,4,6],[2,4,6]],[[1,2,3],[1,2,3],[1,2,3]]])

a2 = np.array([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])

a3 = np.array([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])

b = [np.zeros((3,3)) for i in range(3)]
for i in range(len(b)):
    b[i] = a1[i] + a2[i] + a3[i]

print(b)
