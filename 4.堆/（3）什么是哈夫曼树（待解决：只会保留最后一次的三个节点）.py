'''什么事哈夫曼树'''

'''
假如我想将百分制分数转换为五分制分数，有以下操作
if score<60 ,grade=1;
elif score<70 grade=2;
elif score<80 grade=3;
elif score<90 grade=4;
else grade=5;

这样可以形成一个类似二叉树的判定数，假如我班级所有人都80多分，则每次都需要判断4次，很浪费

那我们可以根据不同成绩出现的频率，通过某种方法重新画树，优化效率
'''

'''
带权路径长度（WPL）：设二叉树有n个叶节点，每个叶节点有一个对应的权值（频率）Wk，从根节点到叶节点的路径长度为Lk
                  则 WPL = ∑Wk*Lk
哈夫曼树（最优二叉树）：WPL值最小的二叉树
'''

'''
哈夫曼树特点：
（1）没有度为1的节点
（2）n个叶节点的哈夫曼树，拥有(2n-1)个节点
（3）任意非叶节点的左右子树交换，仍是哈夫曼树
（4）对同一组权值，有可能存在不同构的两棵哈夫曼树
'''

'''！！！！！！！！！！！！！！！！'''
'''构建哈夫曼树的思路：将叶节点按照权值从小到大顺序排列好，每次调权值最小的两颗二叉树合并为一棵树，最终形成一颗哈夫曼树'''

'''如何做到，每次从中挑选两个最小的呢？？——利用最小堆，每次delete出一个就行,以下为要用到的“最小堆”类'''


class Heap_node:
    def __init__(self,weight,left_son=None,right_bro=None):
        self.weight = weight  # 数据改为权值
        self.left_son = left_son
        self.right_bro = right_bro


class Min_Heap:
    def __init__(self,capacity):  # 初始化最大堆的容量、堆数组、标兵、元素个数
        self.capacity = capacity
        self.elements = [Heap_node(None)]*(capacity+1)
        self.elements[0] = Heap_node(-999999)  # 第一个位置放“哨兵”，树的元素下标从1开始
        self.elements_size = 0  # 元素个数，不算哨兵
        self.head = None  # 为了建树

    def create_heap(self,need_to_heap):  # 方法：针对一组数据，建立最小堆
        # 首先按照顺序随便建立一个二叉树，并存到数组中
        for i in range(len(need_to_heap)):
            self.elements[i+1] = Heap_node(need_to_heap[i])
            self.elements_size += 1

        # 原理：从后往前，依次寻找带儿子的节点，每找到一次进行一次 类似“最小堆删除”的操作，从而达到整个二叉树成为最小堆的效果
        for j in range(self.elements_size-1,0,-1):  # 从最后向前一个一个倒数（不含哨兵），碰到有儿子的节点才进入while循环

            # 从此处开始，与最小堆的删除原理一致
            parent = j
            temporary_data = self.elements[parent].weight

            while parent*2 <= self.elements_size:
                child = parent * 2
                if (child != self.elements_size) and (self.elements[child].weight > self.elements[child+1].weight):
                    child += 1  # 下标替换成右儿子
                if temporary_data < self.elements[child].weight:
                    break
                else:
                    self.elements[parent] = self.elements[child]
                    parent = child
            self.elements[parent] = Heap_node(temporary_data)

    def is_full(self):  # 方法：判断最大堆是否为满
        if self.elements_size < self.capacity:
            return 0
        else:
            return 1

    def is_empty(self):  # 方法：判断最大堆是否为空
        if self.elements_size == 0:
            return 1
        else:
            return 0

    def insert(self,heap_node):  # 方法：最小堆的插入
        if self.is_full():
            print('最小堆已满，无法继续插入元素')
            return

        i = self.elements_size + 1  # 首先默认添加到数组的最后面一位
        # i除以2向下取整，在数组中对应为第i个节点的根节点
        # 当根节点的值小于该节点时，则循环
        while self.elements[i//2].weight > heap_node.weight:  # 此时“哨兵”起了作用，碰到哨兵肯定就退出循环了，i便成了根节点
            self.elements[i] = self.elements[i//2]  # 调换根节点和该节点
            i = i//2
        self.elements[i] = heap_node  # 从最底层一路向上调换，直到合适位置退出循环，此时i为合适的位置下标
        print(f'元素{heap_node.weight}入最小堆成功')
        self.elements_size += 1  # 计数

    def delete_min(self):  # 方法：最小堆的删除
        # 原理：（1）将最后一个元素和第一个根节点元素调换（2）若正确，则不动；否则，一路向下调换直到正确位置
        if self.is_empty():
            print('最小堆已为空，无法继续出元素')
            return

        min_node = self.elements[1]  # 首先将根节点元素值记录下来，方便后续return
        temporary_data = self.elements[self.elements_size].weight  # 记录二叉树的最后一个元素值
        self.elements_size -= 1  # 元素数量-1

        parent = 1  # 首先将最后一个元素放到根节点的位置
        # 以下思路：已知左子树、右子树都是最大堆，我如何调整新的根节点？
        while parent*2 <= self.elements_size:  # 若根节有子树，则循环
            child = parent * 2  # 左儿子在数组中的下标

            # 以下操作：寻找左右儿子中最大的那一个，并让它和temporary比较

            # 如果（左儿子不是最后一个元素）并且（右儿子值 大于 左儿子值）
            if (child != self.elements_size) and (self.elements[child].weight > self.elements[child+1].weight):
                child += 1  # 下标替换成右儿子
            # 如果temporary比自己的左右儿子都大，那么就合乎最大堆的规则，退出循环
            if temporary_data < self.elements[child].weight:
                break
            else:  # 否则，将其替换上来，继续向下扫描
                self.elements[parent] = self.elements[child]
                parent = child

        self.elements[parent] = Heap_node(temporary_data)  # 从最上层一路向下调换，将寻找后的最终位置赋值
        print(f'元素{min_node.weight}出最小堆成功')
        return min_node

    def display(self):  # 方法：显示出完全二叉树结构（后续可以修改成为 建立二叉树方法）
        for s in range(1,self.elements_size+1):
            print(f'节点：{self.elements[s].weight}',end=' -')
            if s*2 <= self.elements_size:
                print(f'左指针：{self.elements[s*2].weight}',end=' -')
            else:
                print(f'左指针：{None}', end=' -')
            if s*2+1 <= self.elements_size:
                print(f'右指针：{self.elements[s*2+1].weight}')
            else:
                print(f'右指针：{None}')

    def struct_tree(self):  # 方法：建立二叉树
        for s in range(1,self.elements_size+1):
            if s == 1:
                self.head = self.elements[s]

            if s*2 <= self.elements_size:
                self.elements[s].left_son = self.elements[s*2]
            else:
                self.elements[s].left_son = None

            if s*2+1 <= self.elements_size:
                self.elements[s].right_bro = self.elements[s*2+1]
            else:
                self.elements[s].right_bro = None


'''以上为要用到的“最小堆”类'''


class Huffman_tree:
    def __init__(self):
        self.head = None

    def build_huffman_tree(self,weight_list):
        min_heap = Min_Heap(len(weight_list))
        min_heap.create_heap(weight_list)
        for i in range(1,min_heap.elements_size):  # size-1次循环（合并）

            temporary_node = Heap_node(-1)

            temporary_node.left_son = min_heap.delete_min()
            temporary_node.right_bro = min_heap.delete_min()
            temporary_node.weight = temporary_node.left_son.weight + temporary_node.right_bro.weight

            min_heap.insert(temporary_node)

        self.head = min_heap.delete_min()

    def pre_order_traversal(self,tree_node):  # 先序遍历（递归方法）
        current_node = tree_node

        if current_node is not None:  # 遍历顺序：先根节点，再左子树，后右子树
            print(f'节点：{current_node.weight}',end=' -')
            if current_node.left_son is not None:
                print(f'左指针：{current_node.left_son.weight}',end=' -')
            else:
                print(f'左指针：{None}', end=' -')
            if current_node.right_bro is not None:
                print(f'右指针：{current_node.right_bro.weight}')
            else:
                print(f'右指针：{None}')
            self.pre_order_traversal(current_node.left_son)
            self.pre_order_traversal(current_node.right_bro)


aa = Huffman_tree()
aa.build_huffman_tree([1,2,3,4,5])
print('------------')
aa.pre_order_traversal(aa.head)  # 待解决问题：不丢失指针
# print(aa.head.weight)
# print(aa.head.left_son.weight)
# print(aa.head.left_son.left_son.weight)

# aa = Min_Heap(8)
# aa.create_heap([9,70,56])
# aa.insert(Heap_node(45))
# aa.insert(Heap_node(141))
# aa.insert(Heap_node(7))
#
# sss = Heap_node(4)
# sss.left_son = Heap_node(31)
# sss.right_bro = Heap_node(54)
# aa.insert(sss)
# print(aa.elements[1].left_son.weight)
#
# sss = Heap_node(3)
# sss.left_son = aa.delete_min()
# sss.right_bro = Heap_node(54)
# aa.insert(sss)
# print(aa.elements[1].left_son.weight)
#
# print('--------')
#
# for i in range(1,aa.elements_size+1):
#     print(aa.elements[i].weight)
#     print(aa.elements[i].left_son)
#
# aa_delete = aa.delete_min()
# print(aa_delete.weight)
# print(aa_delete.left_son.weight)
# print(aa_delete.left_son.left_son.weight)

