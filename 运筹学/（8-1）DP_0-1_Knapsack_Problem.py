"""
0-1背包问题——动态规划方法求解
"""
'''
物品数量 n ，背包容量 b
阶段：以物品数量n为阶段
状态变量：背包容量b
转移方程：fk(λ) = max{fk-1(λ) , ck+fk-1(λ-ak)}
                    不拿物品k , 拿物品k
'''
'''
例：max 10x1 + 7x2 + 25x3 + 24x4
   s.t.
   2x1 + x2 + 6x3 + 5x4 <= 7
   x1,x2,x3,x4 ∈ {0,1}
'''
'''矩阵具体解释可以参考看（8-2）示意图'''


def zero_list(m,n):  # 外部方法：用于建立m*n的 全0 list矩阵
    ll = []
    for i in range(m):
        ll.append([])
        for j in range(n):
            ll[i].append(0)
    return ll


class DP_0_1_Knapsack_Problem:
    def __init__(self,c,a,b):
        self.c = [-1] + c   # 价值系数（第一个值凑数，目的让自变量x从1开始算）
        self.a = [999] + a  # 占用空间（第一个值凑数，目的让自变量x从1开始算）
        self.b = b          # 背包容量

        self.x_num = len(c)  # 自变量数目
        # i行j列代表，考虑第j个物品，此时背包容量为i，此时的最佳选择
        self.state_matrix = zero_list(b+1,self.x_num+1)  # 状态矩阵（第0行0列无意义，但必须有，方便递推）
        # 对应状态矩阵，x的选择状态
        self.x_matrix = zero_list(b+1,self.x_num+1)      # x选择矩阵（第0行0列无意义，但必须有，方便递推）

        self.sol = [0] * (self.x_num + 1)  # 最终解
        self.obj = None  # 最终目标函数值

        self.algorithm()  # 算法主体

    def algorithm(self):  # 方法：算法主体
        for j in range(1,self.x_num+1):  # 对于每一阶段（每个物品代表一个阶段）
            for i in range(1,self.b+1):  # 对于各种背包剩余容量情况

                if i - self.a[j] >= 0:  # 如果 剩余容量-考虑的物品占用空间 还有剩余

                    no_take_j = self.state_matrix[i][j-1]  # 不拿物品j进入背包的情况下的值（直接继承即可）
                    take_j = self.c[j] + self.state_matrix[i-self.a[j]][j-1]  # 拿物品j进入背包的情况下的值（该物品+上一阶段）
                    if no_take_j > take_j:  # 比较哪个更好，更新状态矩阵
                        self.state_matrix[i][j] = no_take_j
                    else:
                        self.state_matrix[i][j] = take_j
                        self.x_matrix[i][j] = 1  # 别忘了同时更新x选择矩阵

                else:  # 如果没有剩余了，不用拿了，继承上一阶段
                    self.state_matrix[i][j] = self.state_matrix[i][j-1]

        self.obj = self.state_matrix[self.b][self.x_num]  # 最终就是最后一行最后一列的值

        capacity = self.b
        x_num = self.x_num
        while x_num >= 0:  # 倒推选择了哪些x
            if self.x_matrix[capacity][x_num] == 1:
                self.sol[x_num] = 1  # 选择了就记录一下
                capacity -= self.a[x_num]  # 同时背包容量 减去 这个选择的物品占用空间

            x_num -= 1  # 直到 倒推回第一个变量

        print(f'最终解为(第一个值是凑数的，无效的，忽视掉)：{self.sol}')
        print(f'最终目标函数值：{self.obj}')


if __name__ == '__main__':
    DP_0_1_Knapsack_Problem([10,7,25,24],[2,1,6,5],7)
