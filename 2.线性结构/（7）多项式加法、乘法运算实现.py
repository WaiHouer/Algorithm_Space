'''多项式相加减的实现算法'''

'''
例子：P1 = 3x^5 + 4x^4 - x^3        + 2x - 1
     P2 =        2x^4 + x^3 - 7x^2 + x
'''


class Node_type:  # 定义多项式的“节点”类
    def __init__(self,coefficient,exponent):
        self.coefficient = coefficient  # 系数
        self.exponent = exponent  # 指数
        self.nxt = None


class P_list:  # 定义“多项式”类，每个节点代表一项
    def __init__(self):
        self.head = None

    def is_empty(self):  # 方法：判断是否为空
        return self.head is None

    def append(self,coefficient,exponent):  # 方法：在多项式末尾加一个节点
        node = Node_type(coefficient,exponent)
        if self.is_empty():  # 先判断是否空列表
            self.head = node  # 是空的，则表头给他
        else:
            current_node = self.head
            while current_node.nxt is not None:   # 不是空的，从头遍历至末尾，然后加上
                current_node = current_node.nxt
            current_node.nxt = node
        # print(f'链表末尾插入元素{coefficient,exponent}成功')

    def travel(self):   # 方法：遍历多项式链表并输出
        current_node = self.head
        i = 1
        while current_node is not None:
            print(f'链表第{i}个元素：',current_node.coefficient,current_node.exponent)
            current_node = current_node.nxt
            i += 1

    def display(self):   # 方法：显示出完整的多项式公式
        current_node = self.head
        print('=',end='')
        while current_node is not None:  # 与遍历同理，只不过输出修改一下
            if current_node.nxt is not None:
                print(f'{current_node.coefficient}X^{current_node.exponent} + ',end='')
            else:
                print(f'{current_node.coefficient}X^{current_node.exponent}')
            current_node = current_node.nxt


def compare(a, b):  # 比较两数字大小函数
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0


'''
加法的思路：两个多项式按照指数递减方式排列好，从头开始比较两项指数大小
大的直接计入结果多项式中，并且指针向后移一项
小的不动
若两者一样，则两者系数相加，系数非0则计入结果多项式中，并且同时指针向后移一项
直到最后，有某一个多项式遍历完了，另一个会剩下点，这时剩下的项全都直接计入结果多项式中
'''


def p_plus(a, b):  # 多项式的加法函数
    p_plus_result = P_list()  # 初始化加法的结果多项式为None
    p1 = a.head
    p2 = b.head

    while p1 is not None and p2 is not None:  # 开始遍历
        if compare(p1.exponent, p2.exponent) == 1:  # 第一个大，直接计入
            p_plus_result.append(p1.coefficient,p1.exponent)
            p1 = p1.nxt  # 指针向后移
        elif compare(p1.exponent, p2.exponent) == -1:  # 第二个大，直接计入
            p_plus_result.append(p2.coefficient,p2.exponent)
            p2 = p2.nxt  # 指针向后移
        else:  # 两者一样，系数相加
            sum_coefficient = p1.coefficient + p2.coefficient
            if sum_coefficient != 0:  # 系数有效，计入结果多项式
                p_plus_result.append(sum_coefficient,p1.exponent)
            p1 = p1.nxt  # 两个指针同时向后移
            p2 = p2.nxt
    # 某个多项式遍历完成了，判断一下，让剩下的直接计入结果多项式
    while p1 is not None:
        p_plus_result.append(p1.coefficient,p1.exponent)
        p1 = p1.nxt

    while p2 is not None:
        p_plus_result.append(p2.coefficient,p2.exponent)
        p2 = p2.nxt

    return p_plus_result


'''
乘法的思路：第一个多项式p1的第一项，与第二个多项式p2的每一项相乘，即：指数相加，系数相乘
          这样就可以形成一个新的多项式，记录下来
          第一个多项式p1的第二项、第三项、......每一次形成一个新的多项式
          最后将这些新多项式用加法运算，加起来，即为最终结果
'''


def p_multiple(a, b):  # 多项式的乘法函数
    p_multiple_result = P_list()  # 初始化加法的结果多项式为None
    p1 = a.head

    while p1 is not None:
        temporary_p = P_list()  # 临时变量，用于存储新增多项式

        p2 = b.head  # 注意：这个头结点赋值位置一定要在第一层循环内，第二层循环外
        while p2 is not None:
            multiple_coefficient = p1.coefficient * p2.coefficient  # 系数相乘
            plus_exponent = p1.exponent + p2.exponent  # 指数相加
            temporary_p.append(multiple_coefficient,plus_exponent)
            p2 = p2.nxt

        p_multiple_result = p_plus(p_multiple_result,temporary_p)  # 新增多项式加法运算
        p1 = p1.nxt

    return p_multiple_result


# p_one = P_list()
# p_one.append(3,5)
# p_one.append(4,4)
# p_one.append(-1,3)
# p_one.append(2,1)
# p_one.append(-1,0)
# p_one.travel()
# print('---------------')
# p_two = P_list()
# p_two.append(2,4)
# p_two.append(1,3)
# p_two.append(-7,2)
# p_two.append(1,1)
# p_two.travel()
# print('---------------')
# p = p_plus(p_one,p_two)
# p.display()


def read(part):  # 读取数据函数：将一行数据读取成为规范的多项式链表结构
    read_result = P_list()
    for i in range(part[0]):
        read_result.append(part[i*2+1],part[i*2+2])  # 行向量中的数字关系，对应着系数和指数
    return read_result


def main():
    part1 = [5,3,5,4,4,-1,3,2,1,-1,0]  # 共5项，P1 = 3x^5 + 4x^4 - x^3        + 2x - 1
    part2 = [4,2,4,1,3,-7,2,1,1]       # 共4项，P2 =        2x^4 + x^3 - 7x^2 + x
    read1 = read(part1)
    read2 = read(part2)
    print('多项式1 ',end=''),read1.display()
    print('多项式2 ',end=''),read2.display()

    p_plus_result = p_plus(read1,read2)
    print('多项式相加结果 ',end=''),p_plus_result.display()

    p_multiple_result = p_multiple(read1,read2)
    print('多项式相乘结果 ',end=''),p_multiple_result.display()


if __name__ == '__main__':
    main()
