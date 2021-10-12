'''二叉搜索树的相关知识'''

'''
三种情况：（1）删除叶节点：直接删除，其父节点指针指向None
        （2）删除只有一个孩子节点的节点：要删除的节点的父节点，指针指向，要删除的节点的孩子节点
        （3）删除左右都有孩子节点的节点：用另外一个节点代替被删除节点（左子树的max元素）或（右子树的min元素），再运用（1）或（2）
            这种方法的好处：（左子树的max元素）或（右子树的min元素）一定不是有两个儿子的节点
                         这样就将问题转换成了（1）（2）情况
'''


class Tree_node:
    def __init__(self, data, left_son=None, right_bro=None):
        self.data = data
        self.left_son = left_son
        self.right_bro = right_bro


class Binary_tree:  # 定义“二叉树”类，其实有没有都行，主要作用：对整个树提供一个指针入口
    def __init__(self, root=None):
        self.root = root

    def is_empty(self):
        return self.root is None

    def traversal_display(self, tree_node):  # 先序遍历（递归方法）
        current_node = tree_node

        if current_node is not None:  # 遍历顺序：先根节点，再左子树，后右子树
            print(current_node.data)
            self.traversal_display(current_node.left_son)
            self.traversal_display(current_node.right_bro)

    def find_min_2(self,tree_node):  # 查找最小元素（非递归方法）
        current_node = tree_node
        if current_node is not None:
            while current_node.left_son:
                current_node = current_node.left_son
            return current_node  # 此处选择返回节点
        else:
            return None

    def find_max_2(self,tree_node):  # 查找最大元素（非递归方法）
        current_node = tree_node
        if current_node is not None:
            while current_node.right_bro:
                current_node = current_node.right_bro
            return current_node  # 此处选择返回节点
        else:
            return None

    def delete(self,tree_node,need_to_delete_data):  # 传入二叉树入口节点，以及要删除的数据，删除方法（递归方法）
        if tree_node is None:  # 递归到None了，还没找到就是没有
            print('元素未找到')
        elif need_to_delete_data < tree_node.data:  # 左右比大小，寻找元素
            # 等式左边 接住 删除完成后的子树，一层一层递归接住，最后返回完整的删除完成后的树
            tree_node.left_son = self.delete(tree_node.left_son,need_to_delete_data)
        elif need_to_delete_data > tree_node.data:
            tree_node.right_bro = self.delete(tree_node.right_bro,need_to_delete_data)

        else:  # 否则，找到了就删掉
            if (tree_node.left_son is not None) and (tree_node.right_bro is not None):  # 如果是拥有两个子树的节点
                right_tree_min = self.find_min_2(tree_node.right_bro)  # 找到右子树最小元素值
                tree_node.data = right_tree_min.data  # 替换
                tree_node.right_bro = self.delete(tree_node.right_bro,right_tree_min.data)  # 删掉原本的 右子树最小元素
            else:
                current_node = tree_node
                if tree_node.left_son is None:  # 如果节点只有 右子树 或 没有子树
                    tree_node = tree_node.right_bro  # 直接这样就行
                else:
                    tree_node = tree_node.left_son
                del current_node  # 不知道有没有用

        return tree_node  # 返回删除完成后的节点


Node_9 = Tree_node(9)
Node_7 = Tree_node(7,None,Node_9)
Node_15 = Tree_node(15)
Node_22 = Tree_node(22)
Node_10 = Tree_node(10,Node_7,Node_15)
Node_20 = Tree_node(20,None,Node_22)
Node_18 = Tree_node(18,Node_10,Node_20)
TREE = Binary_tree(Node_18)

TREE.traversal_display(TREE.root)
print('------------------------')

# TREE.delete(TREE.root,10)  # 两个子节点的节点
# TREE.delete(TREE.root,15)  # 叶节点
TREE.delete(TREE.root,20)  # 一个子节点的节点
TREE.delete(TREE.root,1000)
TREE.traversal_display(TREE.root)
print('------------------------')
