"""
线性回归——梯度下降法
"""
'''具体参考：笔记-线性回归例子-房价预测问题'''
'''有两行小报错，不用管，是QQ输入法造成的'''
import matplotlib.pyplot as plt


class Linear_Regression:  # 线性回归类
    def __init__(self,data):  # 传入训练集数据
        self.data = data         # 训练集
        self.m = len(data['x'])  # 训练集数据个数

        self.alpha = 0.01  # learning rate
        self.sita_0 = 0    # 参数sita_0
        self.sita_1 = 0    # 参数sita_1

        self.iteration = 5000  # 循环次数

        self.algorithm()  # 主函数

    def algorithm(self):  # 方法：主函数
        real_iter = 0  # 记录真实的循环次数
        for i in range(self.iteration):
            # 计算偏导
            tem_0 = self.sita_0 - self.alpha * self.partial_derivative_0()
            tem_1 = self.sita_1 - self.alpha * self.partial_derivative_1()
            # 同步更新参数sita
            if self.sita_0 == tem_0 and self.sita_1 == tem_1:  # 找到最优解退出循环（几乎不可能）
                break
            self.sita_0 = tem_0
            self.sita_1 = tem_1

            real_iter = i
        print(f'回归结果：{self.sita_0, self.sita_1}')
        print(f'实际循环次数：{real_iter}')

        self.display()  # 得到参数值，画图

    def partial_derivative_0(self):  # 方法：计算参数sita_0的偏导数
        result = 0
        for i in range(self.m):
            result += self.sita_0 + self.sita_1 * self.data['x'][i] - self.data['y'][i]
        result *= 1 / self.m
        return result

    def partial_derivative_1(self):  # 方法：计算参数sita_1的偏导数
        result = 0
        for i in range(self.m):
            result += (self.sita_0 + self.sita_1 * self.data['x'][i] - self.data['y'][i]) * self.data['x'][i]
        result *= 1 / self.m
        return result

    def display(self):  # 方法：画图显示
        plt.scatter(self.data['x'], self.data['y'], color="b", label="exam data")  # 散点图
        plt.xlabel("X")  # 添加图标标签
        plt.ylabel("Y")

        x = range(7)
        y = [self.sita_0+self.sita_1*i for i in x]
        plt.plot(x,y,linewidth='1',label="regression",linestyle='-',marker='|')  # 画线
        plt.legend(loc='upper left')  # 图例位置

        plt.show()  # 显示图像


if __name__ == '__main__':
    examDict = {
        'x': [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
              3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50],
        'y': [10, 22, 13, 43, 20, 22, 33, 50, 62,48, 55, 75, 62, 73, 81, 76, 64, 82, 90, 93]
    }
    Linear_Regression(examDict)
