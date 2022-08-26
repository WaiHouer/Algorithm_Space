import numpy as np


# D = np.zeros((len(Arcs),len(inflows)))
# print(D)
# for m in range(1, len(Arcs) + 1):
#     for n in range(1, len(inflows) + 1):  # 外层循环：用来遍历矩阵的每一个元素
#
#         for k in range(0, len(Paths)):  # 内层循环：用来求和
#             if (Arcs[i].id in Paths[j].arcs_code) and
#                 (inflows[i].station1 in Paths[j].stations and inflows[i].station2 in Paths[j].stations):
#                 D[m][n] += X[k]
