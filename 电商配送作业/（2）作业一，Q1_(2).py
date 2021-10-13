from gurobipy import *


class Assignment_Problem:
    def __init__(self):
        self.cij = [[10, 15, 19],
                    [9, 18, 5],
                    [6, 14, 3]]

        # 初始化问题
        self.model = Model('transshipment')
        # 第i个地点向第j个地点运送的货物量
        self.x = self.model.addVars(3, 3, vtype=GRB.BINARY)

        self.algorithm()

    def algorithm(self):
        # 目标函数
        self.model.setObjective(sum(self.x[i, j] * self.cij[i][j] for i in range(3) for j in range(3)), GRB.MINIMIZE)

        # 约束
        self.model.addConstrs(sum(self.x[i, j] for j in range(3)) == 1 for i in range(3))
        self.model.addConstrs(sum(self.x[i, j] for i in range(3)) <= 1 for j in range(3))

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
                print(f'x{i} = ', f'任务{f"{i[0]}"} -> 快递员{f"{i[1]}"}', solution[i])


if __name__ == '__main__':
    Assignment_Problem()
