'''二叉搜索树小题目'''

'''顺序输入十二个月份的英文缩写，自动组合成二叉搜索树'''


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


class Binary_tree:  # 定义“二叉树”类，其实有没有都行，主要作用：对整个树提供一个指针入口
    def __init__(self, root=None):
        self.root = root

    def is_empty(self):
        return self.root is None

    def insert_1(self, tree_node, insert_node_data):  # 寻找合适位置插入元素节点（非递归方法）
        # 注意：此方法只能传入树根！！！
        # 传入树根和要插入的元素值，在合适的位置插入元素节点，并且维持二叉搜索树
        n = 0
        if tree_node is None:
            self.root = Tree_node(insert_node_data)
            print(f'月份{insert_node_data}插入成功')
        else:
            while tree_node:
                if insert_node_data < tree_node.data:
                    if tree_node.left_son is None:
                        n = 1
                        break
                    else:
                        tree_node = tree_node.left_son

                elif insert_node_data > tree_node.data:
                    if tree_node.right_bro is None:
                        n = 2
                        break
                    else:
                        tree_node = tree_node.right_bro

                else:
                    n = 3
                    print(f'二叉树中已经有相同元素{insert_node_data}，无法插入')
                    break
            if n == 1:
                tree_node.left_son = Tree_node(insert_node_data)
                print(f'月份{insert_node_data}插入成功')
            elif n == 2:
                tree_node.right_bro = Tree_node(insert_node_data)
                print(f'月份{insert_node_data}插入成功')

    def traversal_display(self, tree_node):  # 先序遍历（递归方法）
        current_node = tree_node

        if current_node is not None:  # 遍历顺序：先根节点，再左子树，后右子树
            print(current_node.data)
            self.traversal_display(current_node.left_son)
            self.traversal_display(current_node.right_bro)


def main():
    x = ['Jan','Feb','Mar','Apr','May','Jun','July','Aug','Sep','Oct','Nov','Dec']
    Tree = Binary_tree()
    for obj in x:
        Tree.insert_1(Tree.root,obj)

    Tree.traversal_display(Tree.root)
    print(Tree.is_empty())


if __name__ == '__main__':
    main()