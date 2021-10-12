'''可以根据已知的，两个遍历结果，唯一确定一棵树'''

'''tips：已知二叉树的中序遍历结果，和，先序遍历结果 或 后序遍历结果，可以唯一确定一个二叉树'''
pre_list = [1, 2, 3, 4, 5, 6]
in_list = [3,2,4,1,6,5]
post_list = [-1,-1,-1,-1,-1,-1]


def get_post_traversal(pre_first_index,in_first_index,post_first_index,n):  # 函数：已知先序、中序遍历结果，计算得到后序遍历结果
    # 传入三个遍历结果数组的起始点下标（一开始为0，0，0），以及要解决的节点规模
    if n == 0:  # 若没有节点
        return None
    if n == 1:  # 若只有一个节点，后序直接=先序（/中序）
        post_list[post_first_index] = pre_list[pre_first_index]
        return post_list

    root = pre_list[pre_first_index]  # 记录树根的元素值（先序遍历的起始点一定是树根）
    post_list[post_first_index+n-1] = root  # 赋值，后序遍历的末尾一定是树根

    for i in range(n):  # 在中序遍历数组中，寻找树根的下标
        if in_list[in_first_index+i] == root:
            break
    left_n = i  # 则左子树的规模 = i
    right_n = n - left_n - 1  # 则右子树的规模 = n - 左子树规模 - 1（树根）
    # 以下递归解决左、右子树的问题（注意：输入对应的起始点下标）
    # 左子树：先序起始下标--之前先序的起始点+1，中序起始下标--不变，后序起始下标--不变
    get_post_traversal(pre_first_index+1,in_first_index,post_first_index,left_n)
    # 右子树：先序起始下标--之前先序的起始点+左子树元素个数+1，中序起始下标--之前中序的起始点+左子树元素个数+1，后序起始下标--之前后序的起始点+1
    get_post_traversal(pre_first_index+left_n+1,in_first_index+left_n+1,post_first_index+left_n,right_n)


def main():
    get_post_traversal(0,0,0,len(pre_list))
    print(post_list)


if __name__ == '__main__':
    main()
