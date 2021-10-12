'''二叉树的链表存储结构'''


'------------------需要用到堆栈类，在下面规定，此为分割线------------------'


class Node_type:     # 定义“节点”类
    def __init__(self,data):
        self.data = data
        self.nxt = None


class list_stack:    # 定义“链栈”类
    def __init__(self):  # 初始化head
        self.head = None

    def is_empty(self):  # 方法：判断是否为空
        return self.head is None

    def push(self,data):  # 方法：入栈（注意，每次入栈都相当于在队伍head插入一个节点）
        node = Node_type(data)
        node.nxt = self.head
        self.head = node
        # print(f'元素{data}入栈成功')

    def pop(self):  # 方法：出栈（注意，每次出栈都相当于在队伍head输出并删除一个节点）
        if self.is_empty():
            print('堆栈为空，不可出栈')
        else:
            need_to_pop = self.head  # 中间变量记录出栈前的head节点
            self.head = need_to_pop.nxt
            need_to_pop_data = need_to_pop.data  # 交换后，记录出栈节点的数据
            del need_to_pop  # 删除该节点
            # print(f'元素{need_to_pop_data}出栈成功')
            return need_to_pop_data  # 返回数据


'------------------需要用到堆栈类，在上面规定，此为分割线------------------'
'''采用儿子-兄弟表示法，（1）左指针-第一个儿子，（2）右指针-第一个儿子的邻近兄弟'''


class Tree_node:
    def __init__(self, data, left_son=None, right_bro=None):
        self.data = data
        self.left_son = left_son
        self.right_bro = right_bro

    def dict_form(self):  # 节点信息用字典的形式输出
        if (self.left_son is None) and (self.right_bro is None):
            dict_set = {
                '元素': self.data,
                '左指针': None,
                '右指针': None
            }
        elif (self.left_son is not None) and (self.right_bro is None):
            dict_set = {
                '元素': self.data,
                '左指针': self.left_son.data,
                '右指针': None
            }
        elif (self.left_son is None) and (self.right_bro is not None):
            dict_set = {
                '元素': self.data,
                '左指针': None,
                '右指针': self.right_bro.data
            }
        else:
            dict_set = {
                '元素': self.data,
                '左指针': self.left_son.data,
                '右指针': self.right_bro.data
            }
        return dict_set

    def node_size(self):  # 方法：统计包括当前节点在内的，所有子节点数量
        size = 1
        if self.left_son is not None:
            size += self.left_son.node_size()
        if self.right_bro is not None:
            size += self.right_bro.node_size()
        return size


class Binary_tree:  # 定义“二叉树”类，其实有没有都行，主要作用：对整个树提供一个指针入口
    def __init__(self, root=None):
        self.root = root

    def is_empty(self):
        return self.root is None

    def pre_order_traversal_1(self,tree_node):  # 先序遍历（递归方法）
        current_node = tree_node

        if current_node is not None:  # 遍历顺序：先根节点，再左子树，后右子树
            print(current_node.data,'-->',end='')
            self.pre_order_traversal_1(current_node.left_son)
            self.pre_order_traversal_1(current_node.right_bro)

    def in_order_traversal_1(self,tree_node):  # 中序遍历（递归方法）
        current_node = tree_node

        if current_node is not None:  # 遍历顺序：先左子树，再根节点，后右子树
            self.in_order_traversal_1(current_node.left_son)
            print(current_node.data,'-->',end='')
            self.in_order_traversal_1(current_node.right_bro)

    def post_order_traversal_1(self,tree_node):  # 后序遍历（递归方法）
        current_node = tree_node

        if current_node is not None:  # 遍历顺序：先左子树，再右子树，后根节点
            self.post_order_traversal_1(current_node.left_son)
            self.post_order_traversal_1(current_node.right_bro)
            print(current_node.data, '-->', end='')

    def pre_order_traversal_2(self,tree_node):  # 先序遍历（非递归方法）

        stack = list_stack()  # 初始化堆栈（作用：利用堆栈，使节点能到退回去）
        current_node = tree_node  # 临时节点

        # 当（1）树不空（即，没有遍历完），或者（2）堆栈不空（即，没有全部出栈），则一直循环
        while (current_node is not None) or (stack.is_empty() is False):
            # 当树不空时（没走到头）
            while current_node is not None:
                stack.push(current_node)  # 遇到节点就入栈（注意：入栈的是节点，不是节点.data）
                print(current_node.data, '-->', end='')  # 输出
                current_node = current_node.left_son  # 继续左边下一个
            # 当左边走到头了，且堆栈没有全部出栈
            if stack.is_empty() is False:
                # 走到最左侧底部后，开始出栈，并重新赋值给临时节点，让临时节点能够退回到上一个根节点
                current_node = stack.pop()

                # 找到上一个根节点的右侧
                current_node = current_node.right_bro
                # 回到大循环

    def in_order_traversal_2(self,tree_node):  # 中序遍历（非递归方法）
        # 原理：push——顺序其实是先序遍历
        #      pop ——顺序是中序遍历

        stack = list_stack()  # 初始化堆栈（作用：利用堆栈，使节点能到退回去并输出）
        current_node = tree_node  # 临时节点

        # 当（1）树不空（即，没有遍历完），或者（2）堆栈不空（即，没有全部出栈），则一直循环
        while (current_node is not None) or (stack.is_empty() is False):
            # 当树不空时（没走到头）
            while current_node is not None:
                stack.push(current_node)  # 遇到节点就入栈（注意：入栈的是节点，不是节点.data）
                current_node = current_node.left_son  # 继续左边下一个
            # 当左边走到头了，且堆栈没有全部出栈
            if stack.is_empty() is False:
                # 走到最左侧底部后，开始出栈，并重新赋值给临时节点，让临时节点能够退回到上一个根节点
                current_node = stack.pop()
                print(current_node.data,'-->',end='')  # 输出

                # 找到上一个根节点的右侧
                current_node = current_node.right_bro
                # 回到大循环

    # 后序遍历的非递归形式！！！与层序遍历方法的原理一样，只不过一个堆栈，一个队列
    def post_order_traversal_2(self,tree_node):  # 中序遍历（非递归方法）

        stack_1 = []  # 利用两个堆栈进行中序，stack_1是中转堆栈
        stack_2 = []
        stack_1.append(tree_node)  # 先向堆栈1中送入起始点

        # 原理：
        # 中序遍历的顺序是左、右、根，所以要保证stack_2的出栈顺序是左、右、根
        # 所以stack_2的入栈顺序是根、右、左
        # stack_2的入栈元素，靠stack_1出栈来实现，即：stack_1出一个，stack_2入一个
        # 所以stack_1的出栈顺序 == stack_2的入栈顺序，根、右、左

        while stack_1:
            current_node = stack_1.pop()  # （1）根从stack_1出栈//第二次循环：右从stack_1出栈
            stack_2.append(current_node)  # （1）根从stack_2入栈//第二次循环：右从stack_2入栈...

            if current_node.left_son:
                stack_1.append(current_node.left_son)  # （2）先左后右，从stack_1入栈
            if current_node.right_bro:
                stack_1.append(current_node.right_bro)

        while stack_2:  # 所有依次出栈即可
            print(stack_2.pop().data,'-->',end='')


E = Tree_node('E')
H = Tree_node('H')
D = Tree_node('D')
F = Tree_node('F', E)
G = Tree_node('G', None, H)
I = Tree_node('I')
B = Tree_node('B', D, F)
C = Tree_node('C', G, I)
A = Tree_node('A', B, C)
Tree = Binary_tree(A)

print(A.dict_form(),end=''),print(' 节点数为：',A.node_size())
print(B.dict_form(),end=''),print(' 节点数为：',B.node_size())
print(E.dict_form(),end=''),print(' 节点数为：',E.node_size())

print('\n','先序遍历（递归）：')
Tree.pre_order_traversal_1(Tree.root)
print('\n','中序遍历（递归）：')
Tree.in_order_traversal_1(Tree.root)
print('\n','后序遍历（递归）：')
Tree.post_order_traversal_1(Tree.root)
print('\n','---------------------')
print('\n','先序遍历（非递归）：')
Tree.pre_order_traversal_2(Tree.root)
print('\n','中序遍历（非递归）：')
Tree.in_order_traversal_2(Tree.root)
print('\n','后序遍历（非递归）：')
Tree.post_order_traversal_2(Tree.root)
