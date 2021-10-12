'''堆的相关基础知识'''

'''
优先队列：
特殊的“队列”，取出元素的顺序是依照元素的“优先权（关键字）”的大小，而不是进入队列的先后顺序
'''

'''
数组和链表结构来实现“4.堆”，总会有令人不满意的复杂度O(n)，所以考虑使用二叉树
但是二叉树也有问题：每次都找优先权最大的并删掉，会造成树一边倒，不是很平衡

为了解决：让一棵树里面的所有子树的根节点，都是该子树最大的那一个元素，这样删除根节点就行了
“最大堆” = 完全二叉树 + 所有子树根节点为最大元素

堆的两个特性：（1）结构性：用数组表示的完全二叉树
            （2）有序性：任意子树的根节点最大（最小），最大堆（大顶堆）/最小堆（小顶堆）
从根节点到任意节点都是有序的（从小到大或从大到小）
'''

'''！！！！！！！！！！！！！！！！！！！！！！'''
'''用数组表示完全二叉树，所以数组元素可以是单纯的元素值，也可以是节点，以下以节点为例'''


class Heap_node:
    def __init__(self,data,left_son=None,right_bro=None):
        self.data = data
        self.left_son = left_son
        self.right_bro = right_bro


class Max_Heap:
    def __init__(self,capacity):  # 初始化最大堆的容量、堆数组、标兵、元素个数
        self.capacity = capacity
        self.elements = [Heap_node(None)]*(capacity+1)
        self.elements[0] = Heap_node(999999)  # 第一个位置放“哨兵”，树的元素下标从1开始
        self.elements_size = 0  # 元素个数，不算哨兵

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

    def insert(self,data):  # 方法：最大堆的插入
        if self.is_full():
            print('最大堆已满，无法继续插入元素')
            return

        i = self.elements_size + 1  # 首先默认添加到数组的最后面一位
        # i除以2向下取整，在数组中对应为第i个节点的根节点
        # 当根节点的值小于该节点时，则循环
        while self.elements[i//2].data < data:  # 此时“哨兵”起了作用，碰到哨兵肯定就退出循环了，i便成了根节点
            self.elements[i] = self.elements[i//2]  # 调换根节点和该节点
            i = i//2
        self.elements[i] = Heap_node(data)  # 从最底层一路向上调换，直到合适位置退出循环，此时i为合适的位置下标
        print(f'元素{data}入最大堆成功')
        self.elements_size += 1  # 计数

    def delete_max(self):  # 方法：最大堆的删除
        # 原理：（1）将最后一个元素和第一个根节点元素调换（2）若正确，则不动；否则，一路向下调换直到正确位置
        if self.is_empty():
            print('最大堆已为空，无法继续出元素')
            return

        max_data = self.elements[1].data  # 首先将根节点元素值记录下来，方便后续return
        temporary_data = self.elements[self.elements_size].data  # 记录二叉树的最后一个元素值
        self.elements_size -= 1  # 元素数量-1

        parent = 1  # 首先将最后一个元素放到根节点的位置
        # 以下思路：已知左子树、右子树都是最大堆，我如何调整新的根节点？
        while parent*2 <= self.elements_size:  # 若根节有子树，则循环
            child = parent * 2  # 左儿子在数组中的下标

            # 以下操作：寻找左右儿子中最大的那一个，并让它和temporary比较

            # 如果（左儿子不是最后一个元素）并且（右儿子值 大于 左儿子值）
            if (child != self.elements_size) and (self.elements[child].data < self.elements[child+1].data):
                child += 1  # 下标替换成右儿子
            # 如果temporary比自己的左右儿子都大，那么就合乎最大堆的规则，退出循环
            if temporary_data > self.elements[child].data:
                break
            else:  # 否则，将其替换上来，继续向下扫描
                self.elements[parent] = self.elements[child]
                parent = child

        self.elements[parent] = Heap_node(temporary_data)  # 从最上层一路向下调换，将寻找后的最终位置赋值
        print(f'元素{max_data}出最大堆成功')
        return max_data

    def display(self):  # 方法：显示出完全二叉树结构（后续可以修改成为 建立二叉树方法）
        for s in range(1,self.elements_size+1):
            print(f'节点：{self.elements[s].data}',end=' -')
            if s*2 <= self.elements_size:
                print(f'左指针：{self.elements[s*2].data}',end=' -')
            else:
                print(f'左指针：{None}', end=' -')
            if s*2+1 <= self.elements_size:
                print(f'右指针：{self.elements[s*2+1].data}')
            else:
                print(f'右指针：{None}')


aa = Max_Heap(5)
aa.insert(45)
aa.insert(141)
aa.insert(7)
aa.insert(10)
aa.insert(25)
aa.insert(95)
print('---------')
aa.display()
print('---------')
aa.delete_max()
aa.delete_max()
aa.delete_max()
aa.insert(95)
aa.display()

