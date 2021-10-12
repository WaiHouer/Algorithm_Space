'''二叉树的链表层序遍历'''

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

    def leave_pre_traversal(self,tree_node):  # 叶节点遍历（根据先序遍历改的）
        current_node = tree_node

        if current_node is not None:
            # 加一个判断语句，没有左右儿子才输出
            if (current_node.left_son is None) and (current_node.right_bro is None):
                print(current_node.data,'-->',end='')

            self.leave_pre_traversal(current_node.left_son)
            self.leave_pre_traversal(current_node.right_bro)

    def height_post_get(self,tree_node):  # 求二叉树的深度（根据后序遍历改的）
        current_node = tree_node
        # 原理：二叉树高度=max{左子树高度，右子树高度}+1
        if current_node is not None:
            height_left = self.height_post_get(current_node.left_son)
            height_right = self.height_post_get(current_node.right_bro)
            if height_left >= height_right:
                max_height = height_left
            else:
                max_height = height_right
            return max_height+1
        else:
            return 0


'''tips：已知二叉树的中序遍历结果，和，先序遍历结果 或 后序遍历结果，可以唯一确定一个二叉树'''


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
Tree.leave_pre_traversal(Tree.root)

print('\n','二叉树深度：',Tree.height_post_get(Tree.root))
