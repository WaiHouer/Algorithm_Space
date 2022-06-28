"""
作业第三题：动态规划-0-1背包问题

完成时间：2022-5-18

author：@陈典
"""
import math
import numpy as np


class Dynamic_Programming:
    """
    动态规划（逆推法）
    0-1背包问题
    """
    def __init__(self, item_num, capacity, value, weight):
        self.item_num = item_num
        self.capacity = capacity
        self.value = value
        self.weight = weight

        self.stage = self.item_num  # 动态规划阶段数量 = 物品数量

        self.f = 0  # 初始化目标函数值

        self.sk = [i for i in range(self.capacity + 1)]  # sk = 当前阶段剩余背包容量

        self.matrix = []  # 存放最终的计算矩阵
        self.before_node = np.zeros((self.item_num, self.capacity + 1)) + (- 1)  # 存放上一个节点的矩阵，便于逆推

        self.x = [0 for i in range(self.item_num)]  # 存放最终解

        print('特别说明：')
        print('self.before_node 该矩阵用于存放每一节点的上个节点，即来源节点，便于逆推出最终结果')
        print('逆推路线矩阵用于倒推出最优解')
        print('-------------------------')
        print(f'第三题，动态规划-逆推法：')
        print('-------------------------')
        self.algorithm()
        self.solution()  # 逆推出解
        print(f'计算完毕，合并得到完整计算矩阵：')
        for k in range(self.item_num):
            print(f'k = {self.item_num - k}：')
            print(f'0~25： {self.matrix[k][0:math.floor(self.capacity / 2)]}')
            print(f'26~50：{self.matrix[k][math.floor(self.capacity / 2):]}')
        print('-------------------------')
        print('逆推路线矩阵（numpy数组会自动省略过长内容）：')
        print(self.before_node)
        print('-------------------------')
        print(f'最优解：{self.x}')
        print(f'最优函数值：{sum(self.value[i] * self.x[i] for i in range(self.item_num))}')

    def algorithm(self):
        for k in range(self.item_num):
            tem_matrix = [0 for i in range(self.capacity + 1)]  # 临时的计算向量
            print(f'开始计算第{self.item_num - k}阶段')  # 由于逆推，所以阶段倒序

            stage_index = self.item_num - k - 1  # 实际下标倒序

            if k == 0:  # 首先，直接落下来
                for i in range(self.capacity + 1):
                    if self.sk[i] >= self.weight[stage_index]:
                        tem_matrix[i] = self.value[stage_index]

            else:  # 其次，进行比较 max{fk+1(sk), vk + fk+1(sk - wk)}
                for i in range(self.capacity + 1):
                    if self.sk[i] - self.weight[stage_index] >= 0:  # 若上一节点有效，则比较
                        before = self.sk[i] - self.weight[stage_index]

                        if self.matrix[k - 1][i] > self.matrix[k - 1][before] + self.value[stage_index]:
                            tem_matrix[i] = self.matrix[k - 1][i]
                            self.before_node[k][i] = i
                        else:
                            tem_matrix[i] = self.matrix[k - 1][before] + self.value[stage_index]
                            self.before_node[k][i] = before

                    else:
                        tem_matrix[i] = self.matrix[k-1][i]  # 否则直接落下上一阶段决策
                        self.before_node[k][i] = i

            self.matrix.append(tem_matrix)
            print(f'第{self.item_num - k}阶段计算向量（由于过长，所以分段显示）：')
            print(f'0~25 ：{tem_matrix[0:math.floor(self.capacity / 2)]}')
            print(f'26~50：{tem_matrix[math.floor(self.capacity / 2):]}')
            print('-------------------------')

    def solution(self):
        before = [0 for i in range(self.item_num)]
        last_node = -1
        for i in range(self.item_num - 1, -1, -1):  # 逆推出最终路线
            if self.before_node[i][last_node] == -1:
                before[i] = last_node
            else:
                before[i] = self.before_node[i][last_node]
                last_node = int(self.before_node[i][last_node])

        for i in range(self.item_num - 1, -1, -1):  # 根据路线矩阵写出最终解（注意：解是倒序的，需要反转一下）
            if i == self.item_num - 1:
                if self.capacity - before[i] > 0:
                    self.x[i] = 1
                else:
                    self.x[i] = 0
            elif i != 0:
                if before[i + 1] - before[i] > 0:
                    self.x[i] = 1
                else:
                    self.x[i] = 0
            else:
                if before[i] >= self.weight[self.item_num - 1 - i]:
                    self.x[i] = 1

        self.x.reverse()


if __name__ == '__main__':
    i_num = 20
    c = 50
    v = [3, 1, 2, 2, 6, 4, 1, 2, 2, 3, 4, 6, 8, 5, 3, 2, 5, 4, 3, 2]
    w = [2, 6, 2, 4, 3, 9, 2, 2, 3, 3, 3, 4, 4, 4, 5, 3, 2, 4, 4, 3]

    # i_num = 6
    # c = 17
    # v = [3, 1, 2, 2, 6, 4]
    # w = [2, 6, 2, 4, 3, 9]

    Dynamic_Programming(i_num, c, v, w)
