import math
import numpy as np
from gurobipy import *
import pulp as pl

# model = Model('ssss')
# x = model.addVars(3,vtype=GRB.INTEGER)
#
# model.setObjective(2 * x[0] + 3 * x[1] + 4 * x[2], GRB.MAXIMIZE)
#
# model.addConstr(3 * x[0] + 4 * x[1] + 5 * x[2] <= 23, name='')
# model.addConstrs(x[i] >= 0 for i in range(3))
#
# model.optimize()
#
# print(model.ObjVal)
# print(model.getAttr('x'))

ProbLp = pl.LpProblem("ProbLp", sense=pl.LpMaximize)
print(ProbLp.name)
# x1 = pl.LpVariable('x1', lowBound=0, upBound=None, cat='Integer')
# x2 = pl.LpVariable('x2', lowBound=0, upBound=2, cat='integer')
variables = [pl.LpVariable(f'x{i}',lowBound=0,cat=pl.LpInteger) for i in range(3)]

c = [3, 4, 3]
objective = sum([c[i] * variables[i] for i in range(3)])

constraints = []

a1 = [-2, 3, 1]
constraints.append(sum([a1[i] * variables[i] for i in range(3)]) <= 7)
a2 = [4, 1, 2]
constraints.append(sum([a2[i] * variables[i] for i in range(3)]) <= 12)

ProbLp += objective
for i in constraints:
    ProbLp += i


ProbLp.solve()
print("Shan Status:", pl.LpStatus[ProbLp.status])  # 输出求解状态
for v in ProbLp.variables():
    print(v.name, "=", v.varValue)  # 输出每个变量的最优值
print("F(x) =", pl.value(ProbLp.objective))  # 输出最优解的目标函数值

