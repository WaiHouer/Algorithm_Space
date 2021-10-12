"""
分类问题（离散）——二元分类，Logistic回归算法
"""
from openpyxl import load_workbook
from List_0 import build_list
import math
import matplotlib.pyplot as plt
import random


class Logistic_Regression_0_1:
    def __init__(self,x_list,y_list):
        self.x_list = x_list
        self.y_list = y_list
        self.m = len(x_list)
        self.n = len(x_list[0])

        self.sita = build_list(1,self.n,0)

        self.alpha = 0.01
        self.iter = 5000

        self.algorithm()
        self.display()

    def algorithm(self):
        for p in range(self.iter):
            tem = []
            for j in range(self.n):
                tem_x = self.sita[0][j] - self.alpha * self.partial_derivative(j)
                tem.append(tem_x)
            for j in range(self.n):
                self.sita[0][j] = tem[j]
        print(self.sita)

    def partial_derivative(self,para_num):
        result = 0
        for i in range(self.m):
            sita_x = 0
            for j in range(self.n):
                sita_x += self.sita[0][j] * self.x_list[i][j]
            result += (1 / (1 + math.exp(-sita_x)) - self.y_list[i][0]) * self.x_list[i][para_num]
        return result

    def display(self):
        self.x_list = [[row[i] for row in self.x_list] for i in range(len(self.x_list[0]))]
        for i in range(self.m):
            if self.y_list[i][0] == 0:
                plt.scatter(self.x_list[1][i], self.x_list[2][i], color="b", label="exam data")  # 散点图
            else:
                plt.scatter(self.x_list[1][i], self.x_list[2][i], color="r", label="exam data")  # 散点图
        plt.xlabel("X1")  # 添加图标标签
        plt.ylabel("X2")

        x1 = range(40)
        x2 = [self.sita[0][0] / (-self.sita[0][2]) + self.sita[0][1] / (-self.sita[0][2]) * i for i in x1]
        plt.plot(x1, x2, linewidth='1', label="regression", linestyle='-', marker='|')  # 画线

        plt.show()  # 显示图像


if __name__ == '__main__':
    book = load_workbook('diabetes.xlsx')
    sheet = book[f'diabetes']
    x = build_list(50,3,1)
    for k in range(50):
        if k < 25:
            x[k][1] = random.randint(1,20)
            x[k][2] = random.randint(1,5)
        else:
            x[k][1] = random.randint(21,40)
            x[k][2] = random.randint(6,10)
    y = build_list(50,1,0)
    for k in range(50):
        if k < 25:
            y[k][0] = 0
        else:
            y[k][0] = 1

    Logistic_Regression_0_1(x,y)
