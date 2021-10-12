'''建立一棵完全二叉搜素树'''

'''完全二叉搜索树：
（1）满足是一颗完全二叉树
（2）满足是一颗二叉搜索树
'''
'''编写程序：随便输入一组数据，变成完全二叉搜索树'''

'''由于（1）是完全二叉树,不会有空间浪费
       （2）最后需要层序遍历输出结果
       所以“数组”好于“链表”，以下采取数组这种数据结构
'''
import math


def bubble_sort_optimize(a_need_sort):  # 冒泡排序
    for i in range(1, len(a_need_sort)):
        tag = False   # 设立标记，记录这一趟循环是否发生了交换动作
        for j in range(0, len(a_need_sort)-i):
            if a_need_sort[j] > a_need_sort[j+1]:
                temporary = a_need_sort[j]
                a_need_sort[j] = a_need_sort[j+1]
                a_need_sort[j+1] = temporary
                tag = True  # 发生了交换动作，代表这一趟没有浪费
        if tag is False:  # 每一趟判断一次，若没发生交换动作，就直接退出循环
            break
    return a_need_sort


def get_left_number(n):  # 函数：计算左子树元素个数（具体看课件吧。。。）
    # 原理：完全二叉树元素个数N = 前H层完美二叉树元素个数(2^H-1) + 最后余下的元素个数X
    level = math.floor(math.log(n+1,2))
    x = n + 1 - 2**level
    if 2**(level-1) < x:
        x = 2**(level-1)
    left_number = 2**(level-1) - 1 + x
    return left_number


def complete_binary_search_tree(ordered_list,complete_tree,left,right,tree_root_done):
    # 传入：排好序的列表/最终返回数组/当前考虑的树的起始点下标/当前考虑的树的终止点下标/当前考虑的树根，在最终返回数组中的位置

    n = right - left + 1  # 计算当前考虑的树的总元素个数
    if n == 0:  # 若没有元素
        return None

    left_son_number = get_left_number(n)  # 调用函数：得到当前考虑的树的左子树元素个数

    complete_tree[tree_root_done] = ordered_list[left+left_son_number]  # 存入树根

    left_son_root = tree_root_done * 2 + 1  # 左子树树根，在最终返回数组中的位置
    right_son_root = left_son_root + 1      # 右子树树根，在最终返回数组中的位置

    # 递归解决左、右子树问题
    # 左子树：起始点--不变，终止点--起始点+左子树元素个数-1
    complete_binary_search_tree(ordered_list,complete_tree,left,left+left_son_number-1,left_son_root)
    # 右子树：起始点--当前起始点+左子树元素个数+1（树根），终止点--不变
    complete_binary_search_tree(ordered_list,complete_tree,left+left_son_number+1,right,right_son_root)
    return complete_tree


def main():
    A = [4, 1, 9, 5, 0, 6, 7, 2, 8, 3]
    A_ordered = bubble_sort_optimize(A)
    complete_tree = [-1] * len(A)
    tree = complete_binary_search_tree(A_ordered,complete_tree,0,len(A)-1,0)
    print(tree)
    print('树的层序遍历结果： ',end='')
    for i in tree:
        print(i,'-> ',end='')


if __name__ == '__main__':
    main()
