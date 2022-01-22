# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 00:28:48 2022

@author: lenovo/Dell)--(* v *)
"""
from gurobipy import *


class Inventory:
    def __init__(self):

        # D = [ 7944,3190,8670,14500,21300,12250,6150,3100,4150,18350,15100,16030]
        self.P = [21352, 17546, 19236, 23459, 29876, 24165, 20330, 16650, 14850, 24763, 23038, 26205]
        self.r = 432
        self.n = 12
        self.c = 50000

        # Create a new model
        self.model = Model("inventory")

        # Create variables
        self.v = self.model.addVar(vtype=GRB.CONTINUOUS)
        self.x = self.model.addVars(self.n, vtype=GRB.BINARY)

        self.algorithm()

    def algorithm(self):
        # Set objective
        self.model.setObjective(self.v * self.r + sum(self.c * self.x[j] for j in range(self.n)))

        # Add Constraints
        self.model.addConstr(self.x.sum() <= 2)
        for i in range(self.n):
            self.model.addConstr((1 - self.x[i]) * self.P[i] <= self.v)

        # Optimize model
        self.model.optimize()

        # print(model.ObjVal)
        print(f'最优库存量为：{self.v}')
        print(f'租金为：{self.model.Objval}')

if __name__ == '__main__':
    Inventory()
