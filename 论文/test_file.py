
import matplotlib.pyplot as plt
import numpy as np


scale_ls = [0,30,60,90]
index_ls = ['4/13/20','5/13/20','6/13/20','7/13/20']

plt.xticks(scale_ls,index_ls)  ## 可以设置坐标字
plt.title('Average customer flows Number by Weekdays')
plt.show()
