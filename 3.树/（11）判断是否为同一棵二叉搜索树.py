'''是否为同一棵二叉搜索树的小程序'''


class Tree_node:
    def __init__(self, data, left_son=None, right_bro=None):
        self.data = data
        self.left_son = left_son
        self.right_bro = right_bro
        self.flag = 0  # 用于判断两颗二叉树是否相同的标识


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

    def insert_1(self, tree_root, insert_node_data):  # 寻找合适位置插入元素节点（非递归方法）
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


def check(origin_tree_root, k):  # check函数用于比较节点的位置是否相同
    # 原理：
    # 传入基准树根和进行排查的节点数值k，从头开始寻找其位置
    # 若查找路径上的每一个节点，在以前的查找过程中都遇见过，代表其位置是相同的
    # 否则，路径上出现了新的没见过的节点，代表其位置是不同的
    # 节点位置不同，即树不同

    if origin_tree_root.flag == 1:  # 若以前遇见过
        if k < origin_tree_root.data:  # 左右比较，继续寻找其位置
            return check(origin_tree_root.left_son, k)
        elif k > origin_tree_root.data:
            return check(origin_tree_root.right_bro, k)
        else:
            # 若数值相等了，由于flag=1，意味着这个节点在以前扫描过一次，这一次扫描发现仍然与k相等
            # 说明k这个数字，在要判别的序列里面，重复出现了两次，认为无效，返回0
            return 0
    else:  # 若flag=0，意味着该节点在以前没遇见过
        if origin_tree_root.data == k:
            origin_tree_root.flag = 1
            # 没遇见过，但是刚好这个节点数值与k相同，意味着刚好是这个点，认为该点在树中所在位置相同，返回1
            return 1
        else:
            return 0  # 否则，该点在树中位置不同，返回0


def judge(origin_tree_root, tree_list):  # 传入基准树的树根节点和要比较树的序列，judge函数用于判断两棵树是否相同

    if tree_list[0] != origin_tree_root.data:  # 首先比较根节点
        return 33  # 根节点不同，则树不相同，函数返回33
    else:
        origin_tree_root.flag = 1  # 若根节点相同，则令其flag=1，代表遇见过一次了，继续进行其他节点的比较

    for s in range(1, len(tree_list)):  # 其他节点的比较
        if check(origin_tree_root, tree_list[s]) == 0:  # check函数用于比较节点的位置是否相同，一旦有一个不同（0），就代表树不相同
            return 0  # 树由于节点不同，返回0
    return 1  # 树相同，返回1


def reset_flag(origin_tree_root):  # 基准树的flag刷新函数
    if origin_tree_root.left_son is not None:
        reset_flag(origin_tree_root.left_son)
    if origin_tree_root.right_bro is not None:
        reset_flag(origin_tree_root.right_bro)
    origin_tree_root.flag = 0


def main():
    tree: list = []
    tree.append([3, 1, 4, 2])  # 判断基准树

    tree.append([3, 4, 1, 2])  # 以下均为想要跟基准进行判断的树序列

    tree.append([3, 1, 4, 2])

    tree.append([3, 4, 2, 1])

    tree.append([5, 1, 4, 2])

    origin_tree = Binary_tree()  # 建立一个基准树

    for i in tree[0]:  # 默认以第一个树作为基准树，正常建树即可
        if i == tree[0][0]:
            origin_tree = Binary_tree(Tree_node(i))
        else:
            origin_tree.insert_1(origin_tree.root, i)
    origin_tree.traversal_display(origin_tree.root)
    print('-----------')

    for j in range(1, len(tree)):  # 逐个与基准树相互判断，树是否相同（下标从1到最后，象征着除了基准树外的第1到第n棵树）

        j_tree_list = tree[j]  # 取出一棵树的序列
        judge_result = judge(origin_tree.root, j_tree_list)  # judge函数用于判断两棵树是否相同

        if judge_result == 33:
            print(f'树{j}由于树根不同，两颗二叉搜索树不相同')
        elif judge_result == 0:
            print(f'树{j}由于节点不同，两颗二叉搜索树不相同')
        else:
            print(f'树{j}，两颗二叉搜索树相同')

        reset_flag(origin_tree.root)  # 每循环一次，都要重置一次基准树的flag值，以防影响下一序列的排查


if __name__ == '__main__':
    main()
