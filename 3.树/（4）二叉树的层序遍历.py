'''二叉树的链表层序遍历'''


'------------------需要用到队列类，在下面规定，此为分割线------------------'


class Note_type:
    def __init__(self,data):
        self.data = data
        self.nxt = None


class list_queue:
    def __init__(self):
        self.front = None  # 队列开头指针
        self.rear = None   # 队列末尾指针

    def is_empty(self):  # 方法：判断队列是否为空
        return self.front is None

    def add_queue(self, data):  # 方法：入队列（从末尾入队列）
        node = Note_type(data)
        if self.front is None:   # 如果是第一个元素，那么队列开头和尾部都指向这个节点
            self.front = node
            self.rear = node
            # print(f'元素{data}入队列成功')
        else:
            current_node = self.front   # 若不是第一个元素，则遍历，在最后添加节点
            while current_node.nxt is not None:
                current_node = current_node.nxt
            current_node.nxt = node
            self.rear = node
            # print(f'元素{data}入队列成功')

    def delete_queue(self):  # 方法：出队列（从开头出队列）
        if self.is_empty():
            print('队列为空，无法出队列')
        else:
            need_to_delete = self.front
            if self.front == self.rear:  # 如果只有一个节点
                self.front = None
                self.rear = None
            else:
                self.front = self.front.nxt
            need_to_delete_data = need_to_delete.data
            del need_to_delete
            # print(f'元素{need_to_delete_data}出队列成功')
            return need_to_delete_data


'------------------需要用到队列类，在上面规定，此为分割线------------------'
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

    def level_order_traversal_1(self,tree_node):  # 层序遍历
        # 原理：根、左、右顺序入队列，再出队列——>下一层同理....一直循环入队列、出队列

        queue = list_queue()  # 顺序：从上到下，从左到右，一层一层

        if self.is_empty() is True:
            print('空树，不能遍历')
        queue.add_queue(tree_node)  # 在队列中存入二叉树根
        while queue.is_empty() is False:  # 只要队列不空，就一直出队列
            temporary_node = queue.delete_queue()
            print(temporary_node.data, '-->', end='')

            # 出完一个队列后，再依次存入该节点的左、右节点
            if temporary_node.left_son:
                queue.add_queue(temporary_node.left_son)
            if temporary_node.right_bro:
                queue.add_queue(temporary_node.right_bro)
            # 返回循环，出队列形成根、左、右的顺序


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

print('\n','层序遍历：')
Tree.level_order_traversal_1(Tree.root)

