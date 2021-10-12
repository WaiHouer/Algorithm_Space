'''蚁群算法——TSP问题解决'''
import random

import numpy as np
import math


class Ant:   # “蚂蚁”类
    def __init__(self,start,city_num):
        self.city_num = city_num  # 城市数量
        self.start = start        # 该蚂蚁的起始城市
        self.next_access = []     # 该蚂蚁的禁忌表（下一个允许去到的城市集合）

        self.delta_info = 0  # 某次周游完成后的信息素增量
        self.route = []      # 某次周游完成后的路线（解）

        self.initial_access()  # 计算初始禁忌表

    def initial_access(self):  # 方法：初始化禁忌表（表中含有下一次可以选择的所有城市）
        for i in range(self.city_num):
            if i != self.start:
                self.next_access.append(i)

    def refresh(self,num):  # 方法：刷新禁忌表
        del self.next_access[num]  # 在集合中删除某一个城市


class Ant_Colony_Optimization:  # “蚁群算法”类
    def __init__(self,city_loc):
        self.city_loc = city_loc  # 城市坐标
        self.dist_matrix = []     # 距离矩阵
        self.info_matrix = np.zeros((len(city_loc),len(city_loc)))  # 信息素矩阵

        self.alpha = 3  # 信息素重要程度
        self.beta = 5   # 启发因子重要程度
        self.ant_num = len(self.city_loc)  # 蚂蚁数量（这里以城市数量为例）
        self.q = 1    # Q的取值
        self.rho = 0.3  # 旧信息素蒸发速率

        self.final_sol = []  # 最终解
        self.final_dist = 0  # 最终距离最小值

        self.distance()   # 初始化距离矩阵
        self.algorithm()  # 算法主体

    def distance(self):  # 方法：距离矩阵
        distance = np.zeros((len(self.city_loc), len(self.city_loc)))
        for i in range(len(self.city_loc) - 1):
            for j in range(i + 1, len(self.city_loc)):
                distance[i][j] = math.sqrt(
                    (self.city_loc[i][0] - self.city_loc[j][0]) ** 2 + (self.city_loc[i][1] - self.city_loc[j][1]) ** 2)
                distance[j][i] = distance[i][j]  # 假设是来回距离相同的对称矩阵

        self.dist_matrix = distance

    def initial_sol(self):  # 方法：随机生成初始解
        initial_sol = [i for i in range(len(self.city_loc))]
        np.random.shuffle(initial_sol)

        return initial_sol

    def dist_calculate(self,sol):  # 方法：计算某个解的总路线长度
        dist = 0
        for i in range(len(sol)-1):  # 中间点距离累加
            start = sol[i]
            end = sol[i+1]
            dist += self.dist_matrix[start][end]

        start = sol[-1]  # 最后收尾相加，回到原点
        end = sol[0]
        dist += self.dist_matrix[start][end]

        return dist

    def create_ant(self):  # 方法：生成蚂蚁
        ant = []  # 每个元素相当于一个蚂蚁结构体
        for i in range(self.ant_num):  # 按照蚂蚁数量
            ant.append(Ant(i,len(self.city_loc)))

        return ant

    def refresh_access(self,ant,refresh_num):  # 方法：更新禁忌表
        # 从禁忌表中删除某个城市
        ant.refresh(refresh_num)

    def update_info(self,ant):  # 方法：更新信息素矩阵
        # 第一步：所有信息素都需要先蒸发，蒸发速率为rho
        for i in range(len(self.city_loc)):
            for j in range(len(self.city_loc)):
                self.info_matrix[i][j] = self.info_matrix[i][j] * (1 - self.rho)

        # 第二步：按照每只蚂蚁路线，累加信息素增量
        for i in range(self.ant_num):
            for j in range(len(ant[i].route)-1):
                start = ant[i].route[j]
                end = ant[i].route[j+1]
                self.info_matrix[start][end] += ant[i].delta_info

            start = ant[i].route[-1]  # 首尾相连别忘了
            end = ant[i].route[0]
            self.info_matrix[start][end] += ant[i].delta_info

    def algorithm(self):  # 方法：算法主体
        # 生成初始解
        initial_sol = self.initial_sol()

        # 计算初始解路线长度
        initial_dist = self.dist_calculate(initial_sol)

        m = len(self.city_loc)  # m = 城市数量

        # 录入信息素初始量（各边相同）
        for i in range(len(self.city_loc)):
            for j in range(len(self.city_loc)):
                if i != j:
                    self.info_matrix[i][j] = m / initial_dist  # 采用的计算方式：城市数量/初始解路线长度

        T = 100  # 迭代次数
        self.final_dist = initial_dist  # 初始化最终解
        self.final_sol = initial_sol

        # 总循环T次
        for i in range(T):
            # 初始化蚂蚁（注意：每一次大循环都删除一次蚂蚁，初始化一次蚂蚁）
            ant = self.create_ant()

            # 对每一只蚂蚁进行寻找路线操作，总共len(ant)只蚂蚁
            for j in range(len(ant)):
                current_sol = [ant[j].start]  # 首先在当前解中添加起点

                n = len(self.city_loc)

                # 除了起始点，还需要在确定剩下的n-1个点
                for o in range(n-1):
                    prob = []  # 用于存放剩下的每一个城市（即：禁忌表中的城市），被选择的概率

                    p_down = 0  # 公式的分母
                    # 首先，在计算分子之前，先计算分母（循环一次就够了）
                    for kk in range(len(ant[j].next_access)):
                        city_kk = ant[j].next_access[kk]
                        eta_kk = 1 / self.dist_matrix[current_sol[-1]][city_kk]
                        p_down += math.pow(self.info_matrix[current_sol[-1]][city_kk], self.alpha) \
                            * math.pow(eta_kk, self.beta)

                    # 然后，对每个禁忌表中的城市，进行分子的计算，并计算最终的概率
                    for k in range(len(ant[j].next_access)):
                        city = ant[j].next_access[k]

                        eta = 1/self.dist_matrix[current_sol[-1]][city]
                        p_up = math.pow(self.info_matrix[current_sol[-1]][city],self.alpha) * math.pow(eta,self.beta)

                        p = p_up/p_down  # 分子/分母 = 被选择概率

                        prob.append(p)   # 将概率按顺序依次添加到这里面

                    prob_add = []  # 用于存放累加概率
                    for k in range(len(prob)):
                        if k == 0:
                            prob_add.append(prob[k])
                        else:
                            prob_add.append(prob_add[k-1] + prob[k])

                    # 随机数生成
                    r = random.random()

                    # 轮盘赌，选择符合概率的下一座城市
                    for k in range(len(prob_add)):
                        if k == 0 and r < prob_add[k]:
                            current_sol.append(ant[j].next_access[k])
                            refresh_num = k  # 记录想要删除禁忌表中的第k个城市
                        elif prob_add[k-1] < r < prob_add[k]:
                            current_sol.append(ant[j].next_access[k])
                            refresh_num = k

                    # 更新禁忌表，删除选好的城市
                    self.refresh_access(ant[j],refresh_num)

                # 判断优劣，更新解
                current_dist = self.dist_calculate(current_sol)
                if current_dist < self.final_dist:
                    self.final_dist = current_dist
                    self.final_sol = current_sol

                # 录入蚂蚁的信息素增量、该次循环中走过的路线
                ant[j].delta_info = self.q / current_dist
                ant[j].route = current_sol

            # 退出小循环，更新信息素矩阵
            self.update_info(ant)

            del ant  # 本次迭代完成，全部蚂蚁周游完成，死去

        # 以下为用于测试的显示内容
        print(self.dist_matrix)
        # aa = self.dist_matrix.ravel()[np.flatnonzero(self.dist_matrix)]
        # print(min(aa))
        print(self.final_sol)
        aa = 0
        for i in range(len(self.final_sol)-1):
            start = self.final_sol[i]
            end = self.final_sol[i+1]
            print(f'{start}-{end}',self.dist_matrix[start][end])
            aa += self.dist_matrix[start][end]
            print(f'累积和：{aa}')
        print(self.final_dist)


if __name__ == '__main__':
    # city_location = [[0,1],[1,1],[1,0],[2,0],[0,2]]
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

    Ant_Colony_Optimization(city_location)

'''
可改进的地方：
（1）算法主体部分过长，可以拆分成为方法，先这么用着吧
（2）终止条件目前只有到循环次数达到上限才会停止，以后可以加上判断是否出现“停滞现象”
'''

