import random
import numpy as np

total_num = 6

order = [[2, '<', 1],
         [5, '<', 1],
         [1, '<', 4]]  # [2, '<', 6]加上就不对了，不能交叉
print(f'初始父子关系:{order}')

level = {}
for i in range(total_num):
    level[f'{i}'] = []

for i in range(1, total_num + 1):
    tag_left = 1  # 1=左边找到了
    tag_right = 0  # 1=右边找到了
    for tem_order in order:
        if tem_order[0] == i:  # 在左边找到了，则代表不是最高层级
            tag_left = 0
            break

    for tem_order in order:
        if tem_order[2] == i:  # 经过上一步过滤，若在右边找到了，则代表是最高层
            tag_right = 1
            break
    if (tag_left == 1) and (tag_right == 1):
        level['0'].append(i)

for j in range(total_num):  # 对订单拆分层级
    for tem_order in order:
        if (tem_order[2] in level[f'{j}']) and (tem_order[0] not in level[f'{j + 1}']):
            level[f'{j + 1}'].append(tem_order[0])

for i in range(total_num):  # 删除空值
    if not level[f'{i}']:
        del level[f'{i}']
print(f'level:{level}')

order_process = [4,3,3,4,2,3]  # 位置代表订单编号，内容代表订单工序
print(order_process)
print('---------------')

tem_list = [[] for i in range(len(level))]  # 按相对位置，分块编码
for i in range(len(level)):

    for j in range(total_num):
        if j + 1 in level[f'{i}']:
            tem_list[i] = tem_list[i] + [(j + 1) for k in range(order_process[j])]
print(tem_list)

final_coding = []
while tem_list:
    tem = tem_list.pop()
    random.shuffle(tem)  # 打乱编码
    final_coding = final_coding + tem  # 合起来
print(final_coding)

# (1)目前只针对拥有父子关系的订单进行了编码，接下来还需要随机插入没有父子关系的零散订单
# (2)若订单的父子关系有交叉，未能解决（找到最长的关键层级，再基于此确定分叉前后？分支过长的话接着找关键层级？）
# (3)有平行，怎么办

# ——（1）先所有订单完全随机产生一个编码——（2）按照上述方法对各个平行的关系进行生成编码
# ——（3）以（1）确定好的位置，（2）确定好的顺序，将（2）嵌入（1）中
