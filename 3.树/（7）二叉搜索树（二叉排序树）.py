'''二叉搜索树的相关知识'''

'''
性质：（1）非空左子树的所有元素值 < 根节点元素值
     （2）非空右子树的所有元素值 > 根节点元素值
     （3）左、右子树都是二叉搜索树
'''


class Tree_node:
    def __init__(self, data, left_son=None, right_bro=None):
        self.data = data
        self.left_son = left_son
        self.right_bro = right_bro

    def dict_form(self):  # 节点信息用字典的形式输出    !!!!!!!!!!!!有争议哈
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

    def find_1(self,tree_node,x):  # 传入查找的起始节点，以及要查找的元素值（递归方法）
        if tree_node is None:
            return None
        if x > tree_node.data:
            return self.find_1(tree_node.right_bro,x)
        elif x < tree_node.data:
            return self.find_1(tree_node.left_son,x)
        else:
            return tree_node.data

    def find_2(self,tree_node,x):  # 传入查找的起始节点，以及要查找的元素值（非递归方法）
        # 查找的效率 取决于 树的高度（为了保证树不过高，叫做平衡二叉树）
        current_node = tree_node
        while current_node:
            if x > current_node.data:
                current_node = current_node.right_bro
            elif x < current_node.data:
                current_node = current_node.left_son
            else:
                return current_node.data
        return None

    # 最大元素 一定在 最右分枝的端节点
    # 最小元素 一定在 最左分枝的端节点
    def find_min_1(self,tree_node):  # 查找最小元素（递归方法）
        if tree_node is None:
            return None
        elif tree_node.left_son is not None:  # 一路向左，直到走不了了
            return self.find_min_1(tree_node.left_son)
        else:
            return tree_node.data  # 返回节点/节点的data都行

    def find_min_2(self,tree_node):  # 查找最小元素（非递归方法）
        current_node = tree_node
        if current_node is not None:
            while current_node.left_son:  # 一路向左，直到走不了了
                current_node = current_node.left_son
            return current_node.data  # 返回节点/节点的data都行
        else:
            return None

    def find_max_1(self,tree_node):  # 查找最大元素（递归方法）
        if tree_node is None:
            return None
        elif tree_node.right_bro is not None:  # 一路向右，直到走不了了
            return self.find_max_1(tree_node.right_bro)
        else:
            return tree_node.data

    def find_max_2(self,tree_node):  # 查找最大元素（非递归方法）
        current_node = tree_node
        if current_node is not None:
            while current_node.right_bro:  # 一路向右，直到走不了了
                current_node = current_node.right_bro
            return current_node.data
        else:
            return None

    def insert_1(self,tree_root,insert_node_data):  # 寻找合适位置插入元素节点（非递归方法）
        # 注意：此方法只能传入树根！！！
        # 传入树根和要插入的元素值，在合适的位置插入元素节点，并且维持二叉搜索树
        current_node = tree_root
        n = 0  # 初始化标识
        if current_node is None:  # 空树的情况，就建立一个根节点
            # 仅在插入方法中存在的注意点！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            self.root = Tree_node(insert_node_data)  # 注意：千万别用current_node=...，因为在开头的None传进来会造成地址丢失
            print(f'元素{insert_node_data}插入成功')
        else:
            while current_node:
                if insert_node_data < current_node.data:  # 同理，左右比大小找位置
                    if current_node.left_son is None:  # 该位置没有元素，可以插入
                        n = 1  # 标识=1，代表最终插在了left_son上
                        break  # 强制退出循环
                    else:
                        current_node = current_node.left_son

                elif insert_node_data > current_node.data:  # 同理，左右比大小找位置
                    if current_node.right_bro is None:
                        n = 2  # 标识=2，代表最终插在了right_bro上
                        break
                    else:
                        current_node = current_node.right_bro

                else:
                    n = 3  # 标识=3，代表已有元素
                    print(f'二叉树中已经有相同元素{insert_node_data}，无法插入')
                    break
            # 此处根据标识数，判断应该插在哪里
            if n == 1:
                current_node.left_son = Tree_node(insert_node_data)
                print(f'元素{insert_node_data}插入成功')
            elif n == 2:
                current_node.right_bro = Tree_node(insert_node_data)
                print(f'元素{insert_node_data}插入成功')

    def insert_2(self,tree_root,insert_node_data):  # 寻找合适位置插入元素节点（递归方法）
        # 注意：此方法只能传入树根！！！
        current_node = tree_root
        # 仅在插入方法中存在的注意点！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        if self.root is None:  # 并且此条判断，仅在树为空的情况下生效一次，后面递归和它无关了
            self.root = Tree_node(insert_node_data)
            return self.root

        if current_node is None:  # 走到头了，可以插入
            current_node = Tree_node(insert_node_data)
        else:
            if insert_node_data < current_node.data:  # 左右比大小，进行递归
                current_node.left_son = self.insert_2(current_node.left_son,insert_node_data)
            elif insert_node_data > current_node.data:
                current_node.right_bro = self.insert_2(current_node.right_bro,insert_node_data)
            else:
                print(f'二叉树中已经有相同元素{insert_node_data}，无法插入')
        return current_node  # 返回插入结果


Node_9 = Tree_node(9)
Node_7 = Tree_node(7,None,Node_9)
Node_15 = Tree_node(15)
Node_22 = Tree_node(22)
Node_10 = Tree_node(10,Node_7,Node_15)
Node_20 = Tree_node(20,None,Node_22)
Node_18 = Tree_node(18,Node_10,Node_20)
TREE = Binary_tree(Node_18)
TREE_None_1 = Binary_tree()
TREE_None_2 = Binary_tree()

print('递归查找：',TREE.find_1(TREE.root,10))
print('非递归查找：',TREE.find_2(TREE.root,10))
print('非递归查找：',TREE.find_2(TREE.root,333))

print('递归查找最小元素：',TREE.find_min_1(TREE.root))
print('非递归查找最小元素：',TREE.find_min_2(TREE.root))

print('递归查找最大元素：',TREE.find_max_1(TREE.root))
print('非递归查找最大元素：',TREE.find_max_2(TREE.root))
print('---------')
TREE.insert_1(TREE.root,3)
print('非递归插入-查找：',TREE.find_1(TREE.root,7))
TREE.insert_1(TREE.root,21)
print('非递归插入-查找：',TREE.find_1(TREE.root,22))
TREE.insert_1(TREE.root,20)
TREE_None_1.insert_1(TREE_None_1.root,21)
print('非递归插入-空树插入查找：',TREE_None_1.find_1(TREE.root,21))
print('---------')
TREE.insert_2(TREE.root,4)
print('递归插入-查找：',TREE.find_1(TREE.root,3))
TREE.insert_2(TREE.root,19)
print('递归插入-查找：',TREE.find_1(TREE.root,20))
TREE_None_2.insert_2(TREE_None_2.root,21)
print('递归插入-空树插入查找：',TREE_None_2.find_1(TREE.root,21))
print('---------')

TREE.traversal_display(TREE.root)
print('---------')
TREE_None_1.traversal_display(TREE_None_1.root)
print('---------')
TREE_None_2.traversal_display(TREE_None_2.root)
