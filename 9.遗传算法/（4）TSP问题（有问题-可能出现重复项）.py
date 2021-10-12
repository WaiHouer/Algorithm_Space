'''TSP问题——遗传算法解决'''
import numpy as np
import math
import random
'''
TSP问题（旅行商问题）：要拜访N个城市，从某个城市出发，每个城市只经过一次，最后要回到出发的城市
                   保证所选择的路径长度最短
'''


def distance_matrix(city_loc):  # 函数：根据城市坐标，计算距离矩阵
    # 传入：n个城市坐标
    dist = np.zeros((len(city_loc),len(city_loc)))  # 初始化
    # 只计算上三角形部分距离矩阵，下三角形对称赋值即可
    for k1 in range(len(city_loc)-1):
        for k2 in range(k1+1,len(city_loc)):
            dist[k1][k2] = math.sqrt((city_loc[k1][0]-city_loc[k2][0])**2 + (city_loc[k1][1]-city_loc[k2][1])**2)
            dist[k2][k1] = dist[k1][k2]

    return dist  # 返回距离矩阵


def initial_generate(city_num,population):  # 函数：生成初始解
    # 传入：城市数量，种群规模（初始解数量）
    initial_sol = []

    for k in range(population):
        all_city = [k for k in range(city_num)]  # 顺序生成1,2,...,n
        np.random.shuffle(all_city)  # 打乱，得到初始解+1
        initial_sol.append(all_city)

    return initial_sol  # 返回初始解矩阵


def fitness_calculate(one_initial_sol,dist):  # 函数：计算某个解的适应度
    # 传入：某个解，距离矩阵
    fitness = 0
    # 两点之间的距离依次相加，求和
    for k in range(len(one_initial_sol)-1):
        tem_start = one_initial_sol[k]
        tem_end = one_initial_sol[k+1]
        fitness += dist[tem_start][tem_end]

    # 最后再加上首末两点间距离（回到起点）
    start = one_initial_sol[0]
    end = one_initial_sol[len(one_initial_sol)-1]
    fitness += dist[start][end]

    # 由于是最小化为目标，用一个适合的数减去
    fitness = 10 - fitness

    return fitness  # 返回适应度


def selection(initial_sol,dist):  # 函数：选择算子
    # 传入：种群列表，距离矩阵
    fitness = []
    # 计算每个解的适应度
    for k in range(len(initial_sol)):
        fitness.append(fitness_calculate(initial_sol[k],dist))
    # 求和：总适应度
    total_fitness = sum(fitness)

    p_select = [fitness[k] / total_fitness for k in range(len(fitness))]  # 选择概率
    q_select = []  # 累积概率
    for k in range(len(p_select)):
        if k == 0:
            q_select.append(p_select[k])
        else:
            q_select.append(p_select[k] + q_select[k-1])

    selection_result = []  # 存入选择结果
    for k in range(len(q_select)):
        rk = random.random()
        for o in range(len(q_select)):
            if o == 0 and rk < q_select[o]:
                selection_result.append(o)
                break
            elif q_select[o-1] < rk < q_select[o]:
                selection_result.append(o)
                break

    new_solution = []  # 按照选择结果，生成下一代
    for k in selection_result:
        new_solution.append(initial_sol[k])

    return new_solution  # 返回新生代种群


def cross_over(initial_sol):  # 函数：交叉算子
    # 传入：种群列表
    parents = []
    p_cross_over = 0.9
    for k in range(len(initial_sol)):  # 按照交叉概率，选择将要交叉的父辈
        rk = random.random()
        if rk < p_cross_over:
            parents.append(k)
    np.random.shuffle(parents)

    # 判断父辈数量如何进行交叉
    if len(parents) <= 1:
        return
    elif len(parents) % 2 == 1:
        parents.pop()
        while parents:
            parent_1 = parents.pop()
            parent_2 = parents.pop()
            initial_sol[parent_1],initial_sol[parent_2] = \
                cross_over_operate(initial_sol[parent_1],initial_sol[parent_2])  # 进行交叉操作
    else:
        while parents:
            parent_1 = parents.pop()
            parent_2 = parents.pop()
            initial_sol[parent_1],initial_sol[parent_2] = \
                cross_over_operate(initial_sol[parent_1],initial_sol[parent_2])  # 进行交叉操作


def cross_over_operate(parent1,parent2):  # 函数：交叉操作
    # 传入：父辈1，父辈2
    # 在本案例中，我们应用了：部分映射交叉——Partial-Mapped Crossover
    # 具体可以参照（4-1）图片文件

    # 在染色体前半段和后半段，各选取一个随机数作为节点
    loc_1 = random.randint(0,len(parent1) // 2)
    loc_2 = random.randint(len(parent1) // 2 + 1,len(parent1) - 1)

    # 交换两个节点之间的部分（包含两节点）
    tem = parent1
    parent1 = parent1[0:loc_1] + parent2[loc_1:loc_2+1] + parent1[loc_2+1:]
    parent2 = parent2[0:loc_1] + tem[loc_1:loc_2+1] + parent2[loc_2+1:]

    # 交换后，两个父辈的 两个节点之间部分，存入映射矩阵
    # 如：6 9 2 1
    # 如：3 4 5 6
    map_1 = parent1[loc_1:loc_2+1]
    map_2 = parent2[loc_1:loc_2+1]
    remove_list = []  # 移除矩阵（为了最后移除映射表中的中间数）
    # 此时：6->3/9->4/2->5/1->6 （发现有6->3/1->6，实际上为1->3，下面介绍如何消除中间项）

    for k1 in range(len(map_1)):  # 从头依次找两个映射表
        for k2 in range(len(map_1)):
            if map_1[k1] == map_2[k2] and k1 != k2:
                # 如果：两个映射表有相同数字（代表这是中间数），并且这两个数字的下标位置不同（代表此前，并未对此中间数进行处理）
                # 交换某一个映射表中，这两个位置上的元素，目的是将中间数换到相同下标上面去，
                # 方便后面删除（这里是映射表1）
                tem = map_1[k1]
                remove_list.append(tem)  # 添加到移除矩阵
                map_1[k1] = map_1[k2]
                map_1[k2] = tem

    if remove_list:  # 若移除矩阵不空
        final_remove_list = []  # 最终确定的移除矩阵
        # 下面的循环 为了去除 移除矩阵 中的重复项（因为上面对中间数的处理过程中，有可能重复收录同一个中间数）
        for k in remove_list:
            if k not in final_remove_list:
                final_remove_list.append(k)

        # 根据移除矩阵，删除映射表中的所有中间数
        for k in final_remove_list:
            map_1.remove(k)
            map_2.remove(k)

    for k in range(0,loc_1):  # 循环两个父辈的 节点1之前部分，映射数字
        for m1 in range(len(map_1)):
            if map_1[m1] == parent1[k]:
                parent1[k] = map_2[m1]
        for m2 in range(len(map_2)):
            if map_2[m2] == parent2[k]:
                parent2[k] = map_1[m2]

    for k in range(loc_2+1,len(parent1)):  # 循环两个父辈的 节点2之后部分，映射数字
        for m1 in range(len(map_1)):
            if map_1[m1] == parent1[k]:
                parent1[k] = map_2[m1]
        for m2 in range(len(map_2)):
            if map_2[m2] == parent2[k]:
                parent2[k] = map_1[m2]

    return parent1,parent2  # 返回交叉后的父辈


def mutation(initial_sol):  # 函数：变异算子
    # 传入：种群列表
    p_mutation = 0.01
    # 对每条染色体，的每个基因进行扫描，按照变异概率，选择将要变异的节点
    for k in range(len(initial_sol)):
        # 每条基因是个大循环（因此TSP特殊，变异 = 某条染色体上的基因进行交换）
        mutation_list = []
        for m in range(len(initial_sol[0])):
            rk = random.random()
            if rk < p_mutation:
                mutation_list.append(m)
        np.random.shuffle(mutation_list)

        # 判断变异节点数量，进行相关变异操作
        if len(mutation_list) <= 1:
            continue
        elif len(mutation_list) % 2 == 1:
            mutation_list.pop()
            while mutation_list:
                m_1 = mutation_list.pop()
                m_2 = mutation_list.pop()
                # 交换变异节点
                tem = initial_sol[k][m_1]
                initial_sol[k][m_1] = initial_sol[k][m_2]
                initial_sol[k][m_2] = tem
        else:
            while mutation_list:
                m_1 = mutation_list.pop()
                m_2 = mutation_list.pop()
                tem = initial_sol[k][m_1]
                initial_sol[k][m_1] = initial_sol[k][m_2]
                initial_sol[k][m_2] = tem


if __name__ == '__main__':
    # 该算例fit用200000减去
    # city_location = [[6734,1453],[2233,10],[5530,1424],[401,841],
    #                  [3082,1644],[7608,4458],[7573,3716],[7265,1268],
    #                  [6898,1885],[1112,2049],[5468,2606],[5989,2873],
    #                  [4706,2674],[4612,2035],[6347,2683],[6107,669],
    #                  [7611,5184],[7462,3590],[7732,4723],[5900,3561],
    #                  [4483,3369],[6101,1110],[5199,2182],[1633,2809],
    #                  [4307,2322],[675,1006],[7555,4819],[7541,3981],
    #                  [3177,756],[7352,4506],[7545,2801],[3245,3305],
    #                  [6426,3173],[4608,1198],[23,2216],[7248,3779],
    #                  [7762,4595],[7392,2244],[3484,2829],[6271,2135],
    #                  [4985,140],[1916,1569],[7280,4899],[7509,3239],
    #                  [10,2676],[6807,2993],[5185,3258],[3023,1942]]

    # 该算例fit用30000减去
    # city_location = [[11003.611100,42102.500000],[11108.611100,42373.888900],
    #                  [11133.333300,42885.833300],[11155.833300,42712.500000],
    #                  [11183.333300,42933.333300],[11297.500000,42853.333300],
    #                  [11310.277800,42929.444400],[11416.666700,42983.333300],
    #                  [11423.888900,43000.277800],[11438.333300,42057.222200],
    #                  [11461.111100,43252.777800],[11485.555600,43187.222200],
    #                  [11503.055600,42855.277800],[11511.388900,42106.388900],
    #                  [11522.222200,42841.944400],[11569.444400,43136.666700],
    #                  [11583.333300,43150.000000],[11595.000000,43148.055600],
    #                  [11600.000000,43150.000000],[11690.555600,42686.666700],
    #                  [11715.833300,41836.111100],[11751.111100,42814.444400],
    #                  [11770.277800,42651.944400],[11785.277800,42884.444400],
    #                  [11822.777800,42673.611100],[11846.944400,42660.555600],
    #                  [11963.055600,43290.555600],[11973.055600,43026.111100],
    #                  [12058.333300,42195.555600],[12149.444400,42477.500000],
    #                  [12286.944400,43355.555600],[12300.000000,42433.333300],
    #                  [12355.833300,43156.388900],[12363.333300,43189.166700],
    #                  [12372.777800,42711.388900],[12386.666700,43334.722200],
    #                  [12421.666700,42895.555600],[12645.000000,42973.333300]]

    city_location = [[0,1],[1,1],[1,0],[2,0],[0,2]]

    distance = distance_matrix(city_location)  # 距离矩阵

    initial = initial_generate(len(city_location),4)  # 生成随机初始种群，此问题中 不区分 表现型和基因型！！

    T = 10000  # 进化代数
    new = initial  # 初始化新种群
    final_solution_evaluation = 0  # 初始化最终适应度
    final_solution = None  # 初始化最终解
    final_T = None  # 初始化最终迭代次数
    for i in range(T):
        new = selection(new,distance)

        cross_over(new)

        mutation(new)

        fit = []
        for j in range(len(new)):
            fit.append(fitness_calculate(new[j],distance))

        max_fit_index = fit.index(max(fit))
        max_fit = max(fit)
        if max_fit > final_solution_evaluation:
            final_solution = new[max_fit_index]
            final_solution_evaluation = max_fit
            final_T = i + 1

    print(f'最优解是{final_solution}')
    print(f'目标函数最小值为{10 - final_solution_evaluation}')
    print(f'总迭代次数：{T},在第{final_T}次迭代中获得最优解')
