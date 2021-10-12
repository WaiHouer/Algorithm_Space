'''基本遗传算法'''
import math
import numpy as np
import random
'''
三个主要操作算子：选择算子、交叉算子、变异算子
'''

'''
编码：假设取值范围是U1~U2,我们采取长度为k的二进制号码表示，则总共有2^k个不同的编码
     所以递增量就 = (U1-U2)/(2^k-1)
'''

'''例子：
优化问题 —— max f(x1,x2)=21.5+x1*sin(4πx1)+x2*sin(20πx2)
                   s.t.  -3.0 <= x1 <= 12.1
                          4.1 <= x2 <=  5.8
'''


def get_gene_num(low_num,high_num,decimal):  # 函数：确定染色体串的编码长度
    # 传入：下限，上限，要求精度（小数点后几位）
    k = 0
    # 串的长度 取决于 所要求的精度
    # 则编码长度k满足以下公式：2^(k-1) < (上限-下限) * 10^精度 < 2^k - 1
    while True:
        if 2**(k-1) < (high_num-low_num)*(10**decimal) <= (2**k-1):
            break
        else:
            k += 1
            continue
    return k  # 返回该变量需要的染色体串长度


def phenotype_to_genotype(x1,x2):  # 函数：表现型(实际数字) 转换为 基因型(0-1字符串)
    # 传入：两个表现型（实际数字）
    x1_n = get_gene_num(-3.0,12.1,4)  # 获取两变量对应染色体长度
    x2_n = get_gene_num(4.1,5.8,4)

    # 计算变量对应的“十进制数”
    # 根据“基因型(0-1字符串) 转换为 表现型(实际数字)”函数的公式，逆运算出来
    x1_10 = round((x1-(-3.0)) * (2**x1_n-1) / (12.1-(-3.0)))
    x2_10 = round((x2-4.1) * (2**x2_n-1) / (5.8-4.1))  # 记得用round四舍五入（因为伪随机数有一丢丢误差）

    x1_genotype = bin(x1_10)[2:].zfill(x1_n)  # 转换为二进制0-1字符串，记得用zfill，防止缩写
    x2_genotype = bin(x2_10)[2:].zfill(x2_n)

    return x1_genotype + x2_genotype  # 返回整体基因型


def genotype_to_phenotype(genotype):  # 函数：基因型(0-1字符串) 转换为 表现型(实际数字)
    # 传入：某基因型0-1字符串
    x1_n = get_gene_num(-3.0,12.1,4)  # 获取两变量对应染色体长度
    x2_n = get_gene_num(4.1,5.8,4)

    x1_2 = genotype[0:x1_n]  # 将0-1字符串按长度分开（字符串索引左闭右开）
    x2_2 = genotype[x1_n:]

    # 有公式：x(实际数字) = 下限 + x对应十进制数(即-递增次数) * 递增量
    #                                               其中：递增量 = (上限-下限) / (2^染色体长度 - 1)
    x1_phenotype = -3.0 + int(x1_2,2) * (12.1 - (-3.0)) / (2**x1_n-1)
    x2_phenotype = 4.1 + int(x2_2,2) * (5.8-4.1) / (2**x2_n-1)

    return [x1_phenotype,x2_phenotype]  # 返回表现型（两个变量的实际数字）


def evaluation(x_list):  # 函数：计算某表现型的 适应度（写错了，应该是fitness）
    # 传入：某表现型
    # 由于该问题就是目标最大化函数，所以直接用目标函数作为适应度
    evaluate_num = 21.5 + x_list[0] * math.sin(4*math.pi*x_list[0]) \
                   + x_list[1] * math.sin(20*math.pi*x_list[1])
    return evaluate_num


def selection(initial_solution_2,initial_solution_10):  # 函数：遗传算子——选择算子
    # 传入：初始基因型列表，表现型列表
    eval_x = []
    for k in range(len(initial_solution_10)):  # 对每一组计算适应度并记录
        eval_x.append(evaluation(initial_solution_10[k]))
    total_eval = sum(eval_x)  # 求和得出 适应度总和

    # 每个染色体的选择概率——适应度占比
    p_select = [eval_x[k] / total_eval for k in range(len(eval_x))]
    # 每个染色体的累计概率（++++，加到最后=1）
    q_select = []
    for k in range(len(p_select)):
        if k == 0:
            q_select.append(p_select[0])
        else:
            q_select.append(p_select[k] + q_select[k - 1])  # 注意是pk和qk-1

    rand = np.random.random(len(initial_solution_10))  # 生成10个[0,1]区间的随机数
    selection_result = []  # 存放选择的染色体下标
    # 依次比较随机数在哪个区间，Qn < 随机数 < Qm ，则选择Qm进行遗传
    for k in range(len(initial_solution_10)):
        for o in range(len(initial_solution_10)):
            if o == 0 and rand[k] < q_select[o]:  # 注意第一个区间特殊一些，因为j-1=-1
                selection_result.append(o)
                break
            elif q_select[o-1] < rand[k] < q_select[o]:
                selection_result.append(o)
                break

    new_solution_2 = []  # 存入新一代染色体
    for k in selection_result:
        new_solution_2.append(initial_solution_2[k])

    return new_solution_2


def cross_over(initial_solution_2):  # 函数：遗传算子——交叉算子
    # 传入：基因型列表
    parents = []
    p_cross_over = 0.8  # 交叉概率
    for k in range(len(initial_solution_2)):
        # 每访问一个父辈，生成一个随机数，若随机数<交叉概率，则选择该父辈作为交叉的备选
        rk = random.random()
        if rk < p_cross_over:
            parents.append(k)  # parents用来存 可以交叉的备选父辈
    np.random.shuffle(parents)  # 打乱父辈-方便随机配对

    location_max = len(initial_solution_2[0]) - 1  # 交叉点位的上限位置（基因串长度-1）

    if len(parents) <= 1:
        # 若可以进行交叉的父辈数量<=1，则认为不能进行交叉（起码要两个以上）
        return
    elif len(parents) % 2 == 1:
        # 若可以进行交叉的父辈数量 是 奇数，则舍弃一个，凑成偶数
        parents.pop()
        while parents:  # 父辈不空，循环
            parents1 = parents.pop()
            parents2 = parents.pop()
            # 用函数“cross_over_operate”将一对染色体进行交叉操作
            initial_solution_2[parents1],initial_solution_2[parents2] = \
                cross_over_operate(initial_solution_2[parents1],initial_solution_2[parents2],location_max)
    else:
        # 若可以进行交叉的父辈数量 是 偶数，则直接进行
        while parents:
            parents1 = parents.pop()
            parents2 = parents.pop()
            initial_solution_2[parents1],initial_solution_2[parents2] = \
                cross_over_operate(initial_solution_2[parents1],initial_solution_2[parents2],location_max)


def cross_over_operate(parent_1,parent_2,loc_max):  # 函数：交叉算子的一部分——交叉操作
    # 传入：父辈1基因型，父辈2基因型，交叉节点的位置上限
    location = random.randint(0,loc_max)
    # 我们采取“单点交叉”：随机一个交叉点，该点右侧染色体全部交换
    # 位置的设定：可以每一对都一样（写在该函数外面，再传进来）、也可以每一对都重新随机位置（写在该函数内）
    tem = parent_1[location+1:]
    parent_1 = parent_1[0:location+1] + parent_2[location+1:]
    parent_2 = parent_2[0:location+1] + tem

    return parent_1,parent_2


def mutation(initial_solution_2):  # 函数：遗传算子——变异算子
    # 传入：基因型列表
    p_mutation = 0.01  # 变异概率
    # 我们希望种群内所有基因的1%进行变异，所以需要计算总基因长度：染色体数 * 单个染色体的基因长度
    total_gene_number = len(initial_solution_2) * len(initial_solution_2[0])
    for k in range(1,total_gene_number+1):
        # 每访问一个基因节点，生成一个随机数，<变异概率，则进行变异
        rk = random.random()
        if rk < p_mutation:
            chromosome_num = math.ceil(k / len(initial_solution_2[0]))  # 向上取整：第几个染色体(下标从1开始)
            gene_num = k - (chromosome_num - 1) * len(initial_solution_2[0])  # 这条染色体上的第几个基因(下标从1开始)
            # 变异：0->1或1->0
            if initial_solution_2[chromosome_num-1][gene_num-1] == '0':
                initial_solution_2[chromosome_num-1] = initial_solution_2[chromosome_num-1][0:gene_num-1] + '1' \
                                                     + initial_solution_2[chromosome_num-1][gene_num:]
            else:
                initial_solution_2[chromosome_num-1] = initial_solution_2[chromosome_num-1][0:gene_num-1] + '0' \
                                                     + initial_solution_2[chromosome_num-1][gene_num:]


if __name__ == '__main__':
    # 给定初始解-基因型
    initial_2 = ['000001010100101001101111011111110',
                 '001110101110011000000010101001000',
                 '111000111000001000010101001000110',
                 '100110110100101101000000010111001',
                 '000010111101100010001110001101000',
                 '111110101011011000000010110011001',
                 '110100010011111000100110011101101',
                 '001011010100001100010110011001100',
                 '111110001011101100011101000111101',
                 '111101001110101010000010101101010']
    # 初始解-表现型
    initial_10 = []
    for i in range(len(initial_2)):
        initial_10.append(genotype_to_phenotype(initial_2[i]))

    T = 1000  # 进化代数
    new_2 = initial_2  # 初始化新生代
    new_10 = initial_10  # 初始化新生代
    final_solution_evaluation = 0  # 初始化最优解适应度
    final_solution = None  # 初始化最优解
    final_T = None  # 初始化最优解对应进化代数
    for i in range(T):
        new_2 = selection(new_2,new_10)  # 选择（基因型）——注意：只有select过程有返回值，其余均在矩阵上直接进行修改

        cross_over(new_2)  # 交叉（基因型）

        mutation(new_2)  # 变异（基因型）

        # 计算表现型
        for j in range(len(new_2)):
            new_10[j] = genotype_to_phenotype(new_2[j])

        # 计算适应度
        ev = []
        for j in range(len(new_10)):
            ev.append(evaluation(new_10[j]))

        # 如果更好，则记录
        max_ev_index = ev.index(max(ev))
        max_ev = max(ev)
        if max_ev > final_solution_evaluation:
            final_solution = new_10[max_ev_index]
            final_solution_evaluation = max_ev
            final_T = i + 1

    print(f'最优解是：{final_solution}')
    print(f'目标函数最大值为{final_solution_evaluation}')
    print(f'总迭代次数：{T}，在第{final_T}次迭代中获得最优解')
