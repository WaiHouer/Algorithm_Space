"""
作业一，Q3，思考题
"""
'''
参数：
cij , 从i地去j地，花费的费用 （i,j = 0,1,...,6）

决策变量：
xij , 决定从i去j （i,j = 0,1,...,6）
               （其中 i,j=0,1,2,3,4,5,6对应地点A,B,C,D,E,F,G）

模型：
min ∑i,j cij * xij
s.t.
∑j xij - ∑j xji == 1  , i = 0
∑j xij - ∑j xji == 0  , i != 0,n
∑j xij - ∑j xji == -1 , i = n
            xij >= 0  , i,j=0,1,2,3,4,5,6
'''
from gurobipy import GRB
import gurobipy as gp


class Shortest_Route_Problem:
    def __init__(self):
        self.m = 9999
        self.cij = [[0, 7, 9, 18, self.m, self.m, self.m],
                    [self.m, 0, 3, self.m, 5, self.m, self.m],
                    [self.m, 3, 0, self.m, 4, self.m, self.m],
                    [self.m, self.m, self.m, 0, self.m, 3, self.m],
                    [self.m, 5, 4, self.m, 0, 2, 6],
                    [self.m, self.m, self.m, self.m, 2, 0, 3],
                    [self.m, self.m, self.m, self.m, self.m, self.m, 0]]

        self.city_name = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G'}

        # 初始化问题
        self.model = gp.Model('Shortest_Route')
        # 决定i去j（0-1变量）
        self.x = self.model.addVars(7, 7, vtype=GRB.BINARY)

        self.algorithm()

    def algorithm(self):
        # 目标函数
        self.model.setObjective(sum(self.x[i, j] * self.cij[i][j] for i in range(7) for j in range(7)), GRB.MINIMIZE)

        # 约束
        for i in range(6):
            if i == 0:
                self.model.addConstr(sum(self.x[i, j] for j in range(7)) - sum(self.x[j, i] for j in range(7)) == 1,
                                     name='')
            elif i == 6:
                self.model.addConstr(sum(self.x[i, j] for j in range(7)) - sum(self.x[j, i] for j in range(7)) == -1,
                                     name='')
            else:
                self.model.addConstr(sum(self.x[i, j] for j in range(7)) - sum(self.x[j, i] for j in range(7)) == 0,
                                     name='')

        # 求解
        self.model.optimize()

        # 输出
        print(f'最小成本：{self.model.ObjVal}')
        # for var in self.model.getVars():
        #     if var.x > 0:
        #         print(var.varName,var.x)
        # print(self.x)
        solution = self.model.getAttr('x', self.x)  # 字典格式，key为(0,0)格式
        for i in solution:
            if solution[i] > 0:
                print(f'x{i} = ', f'{self.city_name[f"{i[0]}"]} -> {self.city_name[f"{i[1]}"]}', solution[i])


if __name__ == '__main__':
    Shortest_Route_Problem()
