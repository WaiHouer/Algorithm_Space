"""
求解线性规划的单纯形法
"""
'''
以下程序默认变量x1..xn均 >= 0
例1：max z = 2x1 + x2 ; 5x2 <= 15 ; 6x1 + 2x2 <= 24 ; x1 + x2 <= 5 ; x1,x2 >= 0 
'''
import numpy as np
# 不考虑无解解、等等特殊情况
# 不考虑非标准情况，如min，>=，=等
# 不考虑min问题


class Simplex_Method:
    def __init__(self,tag,objective,matrix):
        self.tag = tag              # 标记问题类型
        self.objective = objective  # 目标函数的系数矩阵

        self.C_b_base = np.zeros((len(matrix),2))  # Cb列 和 base列初始化
        self.b = np.zeros((len(matrix),1))         # b列初始化
        for i in range(len(matrix)):
            self.b[i] = matrix[i][2]

        self.standard_matrix = matrix  # 初始化系数矩阵
        self.standard()  # 对系数矩阵进行处理，成为标准化形式

        self.sigma = [0] * len(self.objective)  # 判断数行初始化

        self.solve()  # 求解问题

    def standard(self):  # 方法：标准化
        for i in range(len(self.standard_matrix)):
            if '<=' in self.standard_matrix[i]:  # 当符号是<=时，加一个松弛变量即可
                self.standard_matrix[i][0].append(1)
                self.standard_matrix[i][1] = '=='
                self.objective.append(0)
                self.C_b_base[i][0] = 0
                self.C_b_base[i][1] = f'{len(self.standard_matrix[i][0])}'
                for j in range(len(self.standard_matrix)):
                    if j != i:
                        self.standard_matrix[j][0].append(0)  # 同时要注意别的行加个零（因为某个松弛变量只存在某一行）
            # 另两种情况还没做
            elif '>=' in self.standard_matrix[i]:
                print('还没做好')
            else:
                print('还没做好')

        print('标准化形式为：')
        print(f'ob,{self.objective}')
        print('s.t.')
        for i in range(len(self.standard_matrix)):
            print(self.standard_matrix[i])

    def solve(self):  # 方法：求解
        sigma_tag = 1  # 1代表仍有优化空间，0代表无优化空间

        # 把系数矩阵提取出来
        s_t_matrix = np.zeros((len(self.standard_matrix),len(self.standard_matrix[0][0])))
        for i in range(len(self.standard_matrix)):
            for j in range(len(self.standard_matrix[0][0])):
                s_t_matrix[i][j] = self.standard_matrix[i][0][j]

        # sita值初始化
        sita = np.zeros((len(self.standard_matrix),1))

        while sigma_tag == 1:  # 当有优化空间时，循环

            for i in range(len(self.objective)):  # 计算所有的判断数
                z = 0
                for j in range(len(self.C_b_base)):
                    z += self.C_b_base[j][0] * s_t_matrix[j][i]
                self.sigma[i] = self.objective[i] - z

            sigma_max_index = self.sigma.index(max(self.sigma))  # 找到最大的判断数 的下标（因为是max问题，所以是找最大的正数）

            for i in range(len(sita)):  # 计算每一行的sita值
                if s_t_matrix[i,sigma_max_index] != 0:  # 如果分母不=0，正常算
                    sita[i][0] = self.b[i] / s_t_matrix[i,sigma_max_index]
                else:
                    sita[i][0] = -1  # 如果分母=0，无意义，给个负数就行

            sita_min = 1000000  # 初始化 sita最小值
            bef_index = -1      # 初始化 上一个sita的下标
            aft_index = -1      # 初始化 这一个sita的下标
            sita_min_index = -1  # 初始化 最小sita的下标

            for i in range(len(sita)):
                if 0 <= sita[i][0] < sita_min:  # 如果有效且更好，更新最小
                    sita_min = sita[i][0]
                    sita_min_index = i
                    aft_index = i  # 同时，记本次处理的sita下标为“这一个下标”

                elif sita[i][0] == sita_min:  # 如果相等（退化现象，选择变量Xn下标较小的进行出基操作）
                    index_1 = int(self.C_b_base[bef_index][1])  # 提取base列的Xn下标
                    index_2 = int(self.C_b_base[aft_index][1])
                    # 这一个和上一个比，那个下标小选哪个
                    if index_1 < index_2:
                        sita_min_index = bef_index
                    else:
                        sita_min_index = aft_index

                bef_index = i  # 循环结束，记 本次处理的sita下标为“上一个下标”

            # 更新基变量Cb和base列
            self.C_b_base[sita_min_index][0] = self.objective[sigma_max_index]
            self.C_b_base[sita_min_index][1] = f'{sigma_max_index+1}'

            # 选中的行进行计算
            r = s_t_matrix[sita_min_index][sigma_max_index]
            self.b[sita_min_index] /= r
            for i in range(len(self.objective)):
                s_t_matrix[sita_min_index][i] /= r
            # 其他行进行相应的消元计算
            for i in range(len(sita)):
                if i != sita_min_index:
                    rr = s_t_matrix[i][sigma_max_index] / s_t_matrix[sita_min_index][sigma_max_index]
                    for j in range(len(self.objective)):
                        s_t_matrix[i][j] -= s_t_matrix[sita_min_index][j] * rr
                    self.b[i] -= self.b[sita_min_index] * rr

            # 更新状态，是否还有可优化空间
            sigma_tag = 0
            for i in range(len(self.sigma)):
                if self.sigma[i] > 0:
                    sigma_tag = 1

        ob = 0
        for i in range(len(self.b)):
            ob += self.C_b_base[i][0] * self.b[i]

        print('求解得到最终系数矩阵：')
        print(s_t_matrix)
        print('基变量为：')
        for i in range(len(self.C_b_base)):
            print(f'X{str(int(self.C_b_base[i][1]))} = {self.b[i][0]}')
        print('目标函数值为：')
        print(ob)


if __name__ == '__main__':
    example_1_ob = [100,10,1]
    example_1_matrix = [[[1,0,0],'<=',1],
                        [[20,1,0],'<=',100],
                        [[200,20,1],'<=',10000]]
    Simplex_Method('max',example_1_ob,example_1_matrix)
