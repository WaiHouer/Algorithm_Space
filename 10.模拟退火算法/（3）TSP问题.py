'''TSP问题——模拟退火算法解决'''
import random
import math
import numpy as np

'''
TSP问题（旅行商问题）：要拜访N个城市，从某个城市出发，每个城市只经过一次，最后要回到出发的城市
                   保证所选择的路径长度最短
'''


class Simulated_Annealing:
    def __init__(self, city_loc, tag='min'):
        self.city_loc = city_loc
        self.tag = tag
        self.T_max = 10000
        self.T_min = 1
        self.T_iteration = 1000
        self.rate = 0.95

        self.final_sol = []
        self.final_f = None

        self.dist = self.distance()
        self.algorithm()
        self.display()

    def distance(self):
        distance = np.zeros((len(self.city_loc), len(self.city_loc)))
        for i in range(len(self.city_loc) - 1):
            for j in range(i + 1, len(self.city_loc)):
                distance[i][j] = math.sqrt(
                    (self.city_loc[i][0] - self.city_loc[j][0]) ** 2 + (self.city_loc[i][1] - self.city_loc[j][1]) ** 2)
                distance[j][i] = distance[i][j]

        return distance

    def initial_sol(self):
        city_num = len(self.city_loc)
        initial_sol = [k for k in range(city_num)]
        np.random.shuffle(initial_sol)

        return initial_sol

    def function(self, initial_sol):
        f = 0
        for i in range(len(initial_sol) - 1):
            tem_start = initial_sol[i]
            tem_end = initial_sol[i + 1]
            f += self.dist[tem_start][tem_end]

        end = initial_sol[0]
        start = initial_sol[-1]
        f += self.dist[start][end]

        return f

    def new_sol_f(self, initial_sol, f_old):
        # 通过邻域动作获得邻域内的新解,同时计算出新解的能量值
        # 本案例：随机交换两个相邻点（注意：不触及首末点，方便计算能量值）
        loc_1 = random.randint(1, len(initial_sol) - 3)
        loc_2 = loc_1 + 1
        b_loc_1 = loc_1 - 1
        a_loc_2 = loc_2 + 1

        s1 = initial_sol[b_loc_1]
        s2 = initial_sol[loc_1]
        s3 = initial_sol[loc_2]
        s4 = initial_sol[a_loc_2]

        new_sol = initial_sol[0:loc_1] + [initial_sol[loc_2]] + [initial_sol[loc_1]] + initial_sol[loc_2 + 1:]

        f_new = f_old - self.dist[s1][s2] - self.dist[s2][s3] - self.dist[s3][s4] \
            + self.dist[s1][s3] + self.dist[s3][s2] + self.dist[s2][s4]

        return new_sol,f_new

    def algorithm(self):
        initial_sol = self.initial_sol()

        T = self.T_max
        while T > self.T_min:
            for i in range(self.T_iteration):
                f_old = self.function(initial_sol)

                new_sol,f_new = self.new_sol_f(initial_sol, f_old)

                delta_f = f_new - f_old

                if delta_f < 0:
                    initial_sol = new_sol
                    self.final_sol = new_sol
                    self.final_f = f_new
                else:
                    p = np.exp(-delta_f / T)
                    if random.random() < p:
                        initial_sol = new_sol
                        self.final_sol = new_sol
                        self.final_f = f_new

            T = T * self.rate

    def display(self):
        print(f'初始退火温度：{self.T_max}')
        print(f'退火截止温度：{self.T_min}')
        print(f'同温度迭代次数：{self.T_iteration}')
        print(f'降温速率：{self.rate}')
        print(f'求得最终解为：{self.final_sol}')
        if self.tag == 'min':
            print(f'最小化，目标函数值：{self.final_f}')
        else:
            print(f'最大化，目标函数值：{-self.final_f}')
        # aa = self.dist.ravel()[np.flatnonzero(self.dist)]
        # print(min(aa))
        print(self.dist[17][18])
        print(self.dist[18][16])
        aa = 0
        for i in range(len(self.final_sol)-1):
            start = self.final_sol[i]
            end = self.final_sol[i+1]
            print(f'{start}-{end}',self.dist[start][end])
            aa += self.dist[start][end]
            print(f'累积和：{aa}')


if __name__ == '__main__':
    city_location = [[11003.611100, 42102.500000], [11108.611100, 42373.888900],
                     [11133.333300, 42885.833300], [11155.833300, 42712.500000],
                     [11183.333300, 42933.333300], [11297.500000, 42853.333300],
                     [11310.277800, 42929.444400], [11416.666700, 42983.333300],
                     [11423.888900, 43000.277800], [11438.333300, 42057.222200],
                     [11461.111100, 43252.777800], [11485.555600, 43187.222200],
                     [11503.055600, 42855.277800], [11511.388900, 42106.388900],
                     [11522.222200, 42841.944400], [11569.444400, 43136.666700],
                     [11583.333300, 43150.000000], [11595.000000, 43148.055600],
                     [11600.000000, 43150.000000], [11690.555600, 42686.666700],
                     [11715.833300, 41836.111100], [11751.111100, 42814.444400],
                     [11770.277800, 42651.944400], [11785.277800, 42884.444400],
                     [11822.777800, 42673.611100], [11846.944400, 42660.555600],
                     [11963.055600, 43290.555600], [11973.055600, 43026.111100],
                     [12058.333300, 42195.555600], [12149.444400, 42477.500000],
                     [12286.944400, 43355.555600], [12300.000000, 42433.333300],
                     [12355.833300, 43156.388900], [12363.333300, 43189.166700],
                     [12372.777800, 42711.388900], [12386.666700, 43334.722200],
                     [12421.666700, 42895.555600], [12645.000000, 42973.333300]]

    # city_location = [[0,1],[1,1],[1,0],[2,0],[0,2]]

    Simulated_Annealing(city_location, 'min')
