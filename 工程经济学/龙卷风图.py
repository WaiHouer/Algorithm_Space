"""
工程经济学大作业 —— 龙卷风图代码

日期：2022-5-4

author：@陈典、王锦烨
"""
import numpy as np
from matplotlib import pyplot as plt

variables = ['Operating Income', 'Operating Costs', 'Investment',
             'Working Capital', 'Salvage', 'Recovery of Capital']

base = 7327

change_percent = '10%'

lows = np.array([-35609, -33055, 5464,
                 7250, 7258, 7286])

ups = np.array([50263, 47709, 9189,
                7403, 7395, 7367])

# The y position for each variable
ys = range(len(ups))[::-1]  # top to bottom

for y, low, up in zip(ys, lows, ups):  # 依次画出条形图
    low_width = base - low
    high_width = up - base

    plt.broken_barh(
        [(low, low_width), (base, high_width)],
        (y - 0.4, 0.8), facecolors=['skyblue', 'skyblue'], edgecolors=['darkorange', 'darkorange'],
        linewidth=1.5)

    # 文字显示部分
    x_up = up + 3050
    plt.text(x_up, y, str(up), va='center', ha='center', fontweight='bold')
    plt.text(x_up + 6520, y, f'(+{change_percent})', va='center', ha='center')
    x_low = low - 3050
    plt.text(x_low, y, str(low), va='center', ha='center', fontweight='bold')
    plt.text(x_low - 6520, y, f'(-{change_percent})', va='center', ha='center')

# 龙卷风中心线
plt.axvline(base, color='black')

# 横坐标轴
axes = plt.gca()
axes.spines['left'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_visible(False)
axes.xaxis.set_ticks_position('top')

# 纵坐标轴显示
plt.yticks(ys, variables)

# 显示范围
plt.xlim(min(lows) - 15000, max(ups) + 15000)
plt.ylim(-1, len(variables))
plt.title('Afer-Tax PW')
plt.show()
