from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as mp, random
# 创建样本
a = [199, 188, 170, 157, 118, 99, 69, 44, 22, 1, 5, 9, 15, 21, 30, 40, 50, 60, 70, 79, 88, 97, 99, 98, 70, 46, 39, 33]
for e, y in enumerate((a, [a[i//2]+random.randint(0, 30) for i in range(len(a)*2)])):
    # 待测集
    ly, n = len(y), 2000
    w = [[i / n * 1.2 - .1] for i in range(n)]
    # 建模、拟合、预测
    model = GaussianProcessRegressor()
    model.fit([[i/ly]for i in range(ly)], y)
    z = model.predict(w)
    # 可视化
    mp.subplot(1, 2, e + 1)
    mp.yticks(())
    mp.bar([i/ly for i in range(ly)], y, width=.7/ly)
    mp.scatter(w, z, s=1, color='r')
mp.show()

