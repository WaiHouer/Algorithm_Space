from gurobipy import *


class Transshipment_Problem:
    def __init__(self):
        self.cij = [[9999 for col in range(8)] for row in range(8)]
        self.cij[0][2] = 2; self.cij[0][3] = 3; self.cij[1][2] = 3; self.cij[1][3] = 1; self.cij[2][4] = 2
        self.cij[2][5] = 6; self.cij[2][6] = 3; self.cij[2][7] = 6; self.cij[3][4] = 4; self.cij[3][5] = 4
        self.cij[3][6] = 6; self.cij[3][7] = 5

        self.city_name = {
            '0': '丹佛', '1': '亚特兰大', '2': '堪萨斯城', '3': '路易斯维尔', '4': '底特律', '5': '迈阿密',
            '6': '达拉斯', '7': '新奥尔良'
        }

        # 初始化问题
        self.model = Model('transshipment')
        # 第i个地点向第j个地点运送的货物量
        self.x = self.model.addVars(8, 8, vtype=GRB.CONTINUOUS)

        self.algorithm()

    def algorithm(self):
        # 目标函数
        # obj = 2 * self.x[0,2] + 3 * self.x[0,3] + 3 * self.x[1,2] + self.x[1,3] + 2 * self.x[2,4]
        # + 6 * self.x[2,5] + 3 * self.x[2,6] + 6 * self.x[2,7] + 4 * self.x[3,4] + 4 * self.x[3,5]
        # + 6 * self.x[3,6] + 5 * self.x[3,7]
        # self.model.setObjective(obj,GRB.MINIMIZE)
        self.model.setObjective(sum(self.x[i,j] * self.cij[i][j] for i in range(8) for j in range(8)), GRB.MINIMIZE)

        # 约束
        self.model.addConstr(self.x[0,2] + self.x[0,3] <= 600)
        self.model.addConstr(self.x[1,2] + self.x[1,3] <= 400)
        self.model.addConstr(- self.x[0,2] - self.x[1,2] + self.x[2,4] + self.x[2,5] + self.x[2,6]
                             + self.x[2,7] == 0)
        self.model.addConstr(- self.x[0,3] - self.x[1,3] + self.x[3,4] + self.x[3,5] + self.x[3,6]
                             + self.x[3,7] == 0)
        self.model.addConstr(self.x[2,4] + self.x[3,4] == 200)
        self.model.addConstr(self.x[2,5] + self.x[3,5] == 150)
        self.model.addConstr(self.x[2,6] + self.x[3,6] == 350)
        self.model.addConstr(self.x[2,7] + self.x[3,7] == 300)

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
                print(f'x{i} = ',f'{self.city_name[f"{i[0]}"]} -> {self.city_name[f"{i[1]}"]}',solution[i])


if __name__ == '__main__':
    Transshipment_Problem()
