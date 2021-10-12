'''0-1背包问题：遗传算法解决'''
import random
import numpy as np
import math
'''
已知n个物品的体积size={s1,s2,s3,...,sn},si>0，其价值为value={v1,v2,v3,...,vn},vi>0
假设背包容量为C，哪些物品可以在限制内装入背包，并价值最高呢？
Max f(x1,x2,x3,...,xn)=v1*x1 + v2*x2 +...+ vn*xn
s.t.
s1*x1 + s2*x2 +...+ sn*xn <= C
xi ∈ {0,1}
'''
'''例子：背包容量9，物品体积size={2，3，4，5}，价值value={3，4，5，7}'''


def phenotype_to_genotype(phenotype):  # 函数：表现型(实际数字) 转换为 基因型(0-1字符串)
    # 传入：某表现型的一组变量
    genotype = ''
    for k in range(len(phenotype)):
        genotype += str(phenotype[k])

    return genotype


def genotype_to_phenotype(genotype):  # 函数：基因型(0-1字符串) 转换为 表现型(实际数字)
    # 传入：某基因型0-1字符串
    phenotype = []
    for k in range(len(genotype)):
        phenotype.append(int(genotype[k]))

    return phenotype


def fitness_calculate(phenotype,c,size,value):
    # 传入：某表现型，背包容量，物品大小，物品价值
    total_size = 0
    for k in range(len(phenotype)):
        if phenotype[k] == 1:
            total_size += size[k]

    fitness = 0
    alpha = 3  # 惩罚系数（因为该问题除了决策变量取值范围外，还有其他约束条件，所以需要惩罚）
    for k in range(len(phenotype)):
        fitness += phenotype[k] * value[k]

    # 注意此处：
    # 如果选择的物品总和未超过背包容量上限，则正常计算目标函数值
    # 如果超过了，则 目标函数值 - （超过部分）*惩罚系数
    if total_size <= c:
        return fitness
    else:
        fitness -= alpha * (total_size - c)
        return fitness


def selection(initial_solution_2,initial_solution_10,c,size,value):
    # 传入：基因型列表，表现型列表，背包容量，物品大小，物品价值
    fitness = []
    for k in range(len(initial_solution_10)):
        fitness.append(fitness_calculate(initial_solution_10[k],c,size,value))
    total_fitness = sum(fitness)

    p_elect = [fitness[k] / total_fitness for k in range(len(fitness))]
    q_elect = []
    for k in range(len(p_elect)):
        if k == 0:
            q_elect.append(p_elect[k])
        else:
            q_elect.append(p_elect[k] + q_elect[k-1])

    selection_result = []
    for k in range(len(q_elect)):
        rk = random.random()
        for o in range(len(q_elect)):
            if o == 0 and rk < q_elect[o]:
                selection_result.append(o)
                break
            elif q_elect[o-1] < rk < q_elect[o]:
                selection_result.append(o)
                break

    new_solution_2 = []
    for k in selection_result:
        new_solution_2.append(initial_solution_2[k])

    return new_solution_2


def cross_over(initial_solution_2):
    # 传入：基因型列表
    parents = []
    p_cross_over = 0.9
    for k in range(len(initial_solution_2)):
        rk = random.random()
        if rk < p_cross_over:
            parents.append(k)
    np.random.shuffle(parents)

    location_max = len(initial_solution_2[0]) - 1

    if len(parents) <= 1:
        return
    elif len(parents) % 2 == 1:
        parents.pop()
        while parents:
            parents_1 = parents.pop()
            parents_2 = parents.pop()
            initial_solution_2[parents_1],initial_solution_2[parents_2] = \
                cross_over_operate(initial_solution_2[parents_1],initial_solution_2[parents_2],location_max)
    else:
        while parents:
            parents_1 = parents.pop()
            parents_2 = parents.pop()
            initial_solution_2[parents_1], initial_solution_2[parents_2] = \
                cross_over_operate(initial_solution_2[parents_1], initial_solution_2[parents_2], location_max)


def cross_over_operate(parent1,parent2,loc_max):
    # 传入：父辈1，父辈2，节点位置上限
    location = random.randint(0,loc_max)

    tem = parent1[location+1:]
    parent1 = parent1[0:location+1] + parent2[location+1:]
    parent2 = parent2[0:location+1] + tem

    return parent1,parent2


def mutation(initial_solution_2):
    # 传入：基因型列表
    p_mutation = 0.01
    total_gene_number = len(initial_solution_2) * len(initial_solution_2[0])
    for k in range(1,total_gene_number+1):
        rk = random.random()
        if rk < p_mutation:
            chromosome_num = math.ceil(k / len(initial_solution_2[0]))  # 向上取整：第几个染色体(下标从1开始)
            gene_num = k - (chromosome_num - 1) * len(initial_solution_2[0])  # 这条染色体上的第几个基因(下标从1开始)
            if initial_solution_2[chromosome_num-1][gene_num-1] == '0':
                initial_solution_2[chromosome_num-1] = initial_solution_2[chromosome_num-1][0:gene_num-1] + '1' + \
                                                       initial_solution_2[chromosome_num-1][gene_num:]
            else:
                initial_solution_2[chromosome_num-1] = initial_solution_2[chromosome_num-1][0:gene_num-1] + '0' + \
                                                       initial_solution_2[chromosome_num-1][gene_num:]


if __name__ == '__main__':
    pack_c = 9
    item_size = [2,3,4,5]
    item_value = [3,4,5,7]

    initial_2 = ['1111',
                 '1010',
                 '1001',
                 '1000']  # 1010代表  1，3选择  2，4不选择
    initial_10 = []
    for i in range(len(initial_2)):
        initial_10.append(genotype_to_phenotype(initial_2[i]))

    T = 100
    new_2 = initial_2
    new_10 = initial_10
    final_solution_evaluation = 0
    final_solution = None
    final_T = None
    for i in range(T):
        new_2 = selection(new_2,new_10,pack_c,item_size,item_value)

        cross_over(new_2)

        mutation(new_2)

        for j in range(len(new_2)):
            new_10[j] = genotype_to_phenotype(new_2[j])

        fit = []
        for j in range(len(new_2)):
            fit.append(fitness_calculate(new_10[j],pack_c,item_size,item_value))

        max_fit_index = fit.index(max(fit))
        max_fit = max(fit)
        if max_fit > final_solution_evaluation:
            final_solution = new_10[max_fit_index]
            final_solution_evaluation = max_fit
            final_T = i + 1

    print(f'最优解是：{final_solution}')
    print(f'目标函数最大值为{final_solution_evaluation}')
    print(f'总迭代次数：{T}，在第{final_T}次迭代中获得最优解')
