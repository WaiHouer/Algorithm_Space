"""
作业一，Q2
"""
'''
参数：
cij , 教授i去教授课程j获得的评分 （i,j = 0,1,2,3）

决策变量：
xij , 教授i去教授课程j （i,j = 0,1,2,3）
                    （其中 i=0,1,2,3对应教授A,B,C,D；j=0,1,2,3对应课程UG,MBA,MS,Ph.D）

模型：
max ∑i,j cij * xij
s.t.
∑j xij == 1 , i=0,1,2,3
∑i xij == 1 , j=0,1,2,3
   xij >= 0 , i,j=0,1,2,3
'''
# from gurobipy import *
from gurobipy import GRB
import gurobipy as gp


class Assignment_Problem:
    def __init__(self):
        self.cij = [[2.8, 2.2, 3.3, 3.0],
                    [3.2, 3.0, 3.6, 3.6],
                    [3.3, 3.2, 2.5, 3.5],
                    [3.2, 2.8, 2.5, -99]]

        self.pro_name = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
        self.class_name = {'0': 'UG', '1': 'MBA', '2': 'MS', '3': 'Ph.D'}

        # 初始化问题
        self.model = gp.Model('transshipment')
        # 教授i去教授课程j（0-1变量）
        self.x = self.model.addVars(4, 4, vtype=GRB.BINARY)

        self.algorithm()

    def algorithm(self):
        # 目标函数
        self.model.setObjective(sum(self.x[i,j] * self.cij[i][j] for i in range(4) for j in range(4)), GRB.MAXIMIZE)

        # 约束
        self.model.addConstrs(sum(self.x[i, j] for j in range(4)) == 1 for i in range(4))
        self.model.addConstrs(sum(self.x[i, j] for i in range(4)) == 1 for j in range(4))

        # 求解
        self.model.optimize()

        # 输出
        print(f'最小成本：{self.model.ObjVal}')
        # for var in self.model.getVars():
        #     if var.x > 0:
        #         print(var.varName,var.x)
        # print(self.x)
        solution = self.model.getAttr('x',self.x)  # 字典格式，key为(0,0)格式
        for i in solution:
            if solution[i] > 0:
                print(f'x{i} = ',f'教授{self.pro_name[f"{i[0]}"]} -> 课程{self.class_name[f"{i[1]}"]}',solution[i])


if __name__ == '__main__':
    Assignment_Problem()
