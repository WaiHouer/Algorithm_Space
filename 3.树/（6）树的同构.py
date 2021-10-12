'''二叉树的同构判断'''

'''同构：左右子树可以交换，但每个节点元素data值必须相同，每个节点的子树必须相同'''


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


def isomorphism(tree_1_root,tree_2_root):  # 传入两棵树的树根，判断是否同构（原理很简单，一层一层判断）
    # （1）两树的树根为空，同构
    if (tree_1_root is None) and (tree_2_root is None):
        return 1
    # （2）两树其中之一，树根为空，不同构
    if ((tree_1_root is None) and (tree_2_root is not None)) or ((tree_1_root is not None) and (tree_2_root is None)):
        return 0
    # （3）两树的树根data不相等，不同构
    if tree_1_root.data != tree_2_root.data:
        return 0
    # （4）两树左子树为空，那么继续判断两树的右子树
    if (tree_1_root.left_son is None) and (tree_2_root.left_son is None):
        return isomorphism(tree_1_root.right_bro,tree_2_root.right_bro)
    # （5）两树左子树不为空，且左子树所含data值相等，那么继续同时判断两树的左、右子树
    if ((tree_1_root.left_son is not None) and (tree_2_root.left_son is not None)) and \
            (tree_1_root.left_son.data == tree_2_root.left_son.data):
        return isomorphism(tree_1_root.left_son, tree_2_root.left_son) and \
               isomorphism(tree_1_root.right_bro, tree_2_root.right_bro)
    # （6）否则，那么继续同时判断两树的左/右、右/左子树（即：交换左右子树的情况）
    else:
        return isomorphism(tree_1_root.left_son, tree_2_root.right_bro) and \
               isomorphism(tree_1_root.right_bro,tree_2_root.left_son)


def main():
    F1 = Tree_node('F')
    H1 = Tree_node('H')
    D1 = Tree_node('D')
    E1 = Tree_node('E',F1)
    G1 = Tree_node('G',H1)
    B1 = Tree_node('B',D1,E1)
    C1 = Tree_node('C',G1)
    A1 = Tree_node('A',B1,C1)
    Tree1 = Binary_tree(A1)

    H2 = Tree_node('H')
    F2 = Tree_node('F')
    G2 = Tree_node('G',None,H2)
    E2 = Tree_node('E',F2)
    D2 = Tree_node('D')
    C2 = Tree_node('C',G2)
    B2 = Tree_node('B',E2,D2)
    A2 = Tree_node('A',C2,B2)
    Tree2 = Binary_tree(A2)

    H3 = Tree_node('H')
    F3 = Tree_node('F')
    G3 = Tree_node('G',None,H3)
    E3 = Tree_node('E')
    D3 = Tree_node('D',F3)
    C3 = Tree_node('C',D3,E3)
    B3 = Tree_node('B',G3)
    A3 = Tree_node('A',B3,C3)
    Tree3 = Binary_tree(A3)

    if isomorphism(Tree1.root,Tree2.root) == 0:
        print('Tree1 和 Tree2 两个二叉树 不同构')
    else:
        print('Tree1 和 Tree2 两个二叉树 同构')

    if isomorphism(Tree1.root,Tree3.root) == 0:
        print('Tree1 和 Tree3 两个二叉树 不同构')
    else:
        print('Tree1 和 Tree3 两个二叉树 同构')


if __name__ == '__main__':
    main()